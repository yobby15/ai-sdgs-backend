from app.core import input_doc, retrieval
from app.database import supabase_service, vdb_utils
from app.infrastructure import embedding_service, llm_agent_service, prompt_agent
from app.utils import text_processing

from langchain_community.vectorstores import SupabaseVectorStore

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Literal, Optional
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import time
import json
import logging
import tempfile
import shutil
import functools

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
    encoding="utf-8"
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("hpack").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

#SAVE_PATH = os.getenv("RESULT_SAVE_PATH", "./ai_result")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
MAX_FILE_SIZE = 10 * 1024 * 1024

class AnalyzeResponse(BaseModel):
    id_request: str
    timestamp: str
    model_name: str
    time_execution: float
    result: dict | list | str | None


app_state: dict = {
    "jobs": {}
}


@functools.lru_cache(maxsize=8)
def get_llm(model_name: str, type_api: str):
    """Cache LLM per kombinasi model_name + type_api agar tidak diinisiasi ulang tiap request."""
    logger.info("Inisiasi LLM baru: %s (%s)", model_name, type_api)
    return llm_agent_service.model_init(model_name=model_name, type_api=type_api)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inisiasi resource berat saat startup, cleanup saat shutdown."""
    logger.info("=== [STARTUP] Inisiasi services ===")

    app_state["embeddings"] = embedding_service.embedding_init(model_name="microsoft/harrier-oss-v1-0.6b", type_run="huggingface_inference")
    app_state["supabase"] = supabase_service.supabase_init(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_service_key=os.getenv("SUPABASE_SERVICE_KEY"),
    )
    app_state["supabase_vdb"] = supabase_service.supabase_vdb_init(
        supabase=app_state["supabase"],
        embeddings=app_state["embeddings"]
    )
    app_state["chat_prompt"] = prompt_agent.FULL_CHAT_PROMPT

    logger.info("=== [STARTUP] Services siap ===")
    yield
    logger.info("=== [SHUTDOWN] Cleanup selesai ===")


app = FastAPI(
    title="AI SDG Scoring API",
    description="Analisis dokumen PDF dan cocokkan dengan indikator SDG menggunakan LLM.",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def analyze_document(
        path_file: str,
        source: str,
        supabase_vdb: SupabaseVectorStore,
        llm,
        chat_prompt,
        k: int = 10,
        save_result: bool = True,
) -> dict:
    logger.debug("[START] : Analyzing document")
    model_name = llm.model_dump()['name']
    id_request = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    start_time = time.time()

    extraction_result = input_doc.input_document(path_file=path_file, source=source)
    if isinstance(extraction_result, dict):
        retrieval_result = retrieval.metadata_retrival_SDG(
            metadata_article=extraction_result,
            vdb=supabase_vdb,
            k=k
        )
    elif isinstance(extraction_result, list):
        retrieval_result = retrieval.chunks_retrieval_SDG(
            documents=extraction_result,
            vdb=supabase_vdb,
            k=k
        )
    else:
        raise ValueError("Hasil ekstraksi dokumen tidak valid.")

    prompt_value = chat_prompt.format_prompt(text=retrieval_result)
    full_prompt = prompt_value.to_string()

    logger.info("Send prompt to LLM")
    llm_output = None
    llm_execution_status = False
    try:
        logger.debug("[START] : LLM Agent Analyzing (total chars in prompt:%s)", len(full_prompt))
        llm_output = llm.invoke(prompt_value)
        logger.debug("[END] : LLM Agent Analyzing")
        llm_execution_status = True
    except Exception as e:
        logger.error("Terjadi error pada LLM\n\n%s", e)
        raise

    final_result = text_processing.repair_llm_json(llm_output.content) if llm_execution_status else None
    time_execution = time.time() - start_time

    json_result = {
        "id_request": id_request,
        "timestamp": timestamp,
        "model_name": model_name,
        "time_execution": time_execution,
        "input_model": full_prompt,
        "output_model": llm_output.model_dump() if llm_execution_status else None,
        "result": final_result
    }

    # if save_result:
    #     safe_source = Path(source).stem.replace("/", "_").replace("\\", "_")
    #     folder = Path(SAVE_PATH)
    #     folder.mkdir(parents=True, exist_ok=True)
    #     filename = f"result_{safe_source}_{id_request}.json"
    #     with open(folder / filename, "w", encoding="utf-8") as f:
    #         json.dump(json_result, f, indent=4, ensure_ascii=False)
    #     logger.info("Hasil disimpan ke: %s", folder / filename)

    logger.debug("[END] : Analyzing document")
    return json_result


def process_analysis_task(
        job_id: str, 
        tmp_path: str, 
        tmp_dir: str, 
        doc_source: str, 
        supabase_vdb, 
        llm, 
        chat_prompt, 
        k: int, 
        save_result: bool
):
    """Fungsi yang berjalan di background untuk memproses dokumen."""
    app_state["jobs"][job_id] = {"status": "processing", "result": None, "error": None}
    
    try:
        logger.info(f"Memulai job {job_id} untuk dokumen: {doc_source}")
        result = analyze_document(
            path_file=tmp_path,
            source=doc_source,
            supabase_vdb=supabase_vdb,
            llm=llm,
            chat_prompt=chat_prompt,
            k=k,
            save_result=save_result,
        )
        app_state["jobs"][job_id] = {"status": "completed", "result": result, "error": None}
        logger.info(f"Job {job_id} selesai (%.2fs)", result["time_execution"])

    except Exception as e:
        logger.exception(f"Error pada job {job_id}: {str(e)}")
        app_state["jobs"][job_id] = {"status": "failed", "result": None, "error": str(e)}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def add_sdg_knowledge(
        path_file: str,
        source: str,
        supabase_vdb: SupabaseVectorStore,
        page_range: list[int] = None,
        special_page: list[int] = None
):
    if special_page is None:
        special_page = []

    logger.debug("[START] : Adding SDG Knowledge")
    start_time = time.time()
    extraction_result = input_doc.input_document(
        path_file=path_file,
        source=source,
        type_doc="sdg_knowledge",
        page_range=page_range,
        special_page=special_page
    )
    added_data_ids = vdb_utils.add_data_to_vdb(
        vdb=supabase_vdb,
        documents=extraction_result
    )
    logger.debug("[END] : Adding SDG Knowledge (elapsed: %.2fs)", time.time() - start_time)
    return added_data_ids


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "AI SDG Scoring API berjalan."}


@app.get("/health", tags=["Health"])
def health():
    services_ready = all(k in app_state for k in ["embeddings", "supabase", "supabase_vdb"])
    return {
        "status": "healthy" if services_ready else "degraded",
        "services": {
            "embeddings": "ready" if "embeddings" in app_state else "not_initialized",
            "supabase": "ready" if "supabase" in app_state else "not_initialized",
            "supabase_vdb": "ready" if "supabase_vdb" in app_state else "not_initialized",
        }
    }


@app.post("/analyze", tags=["Analisis"])
async def analyze_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="File PDF yang akan dianalisis (Max 10MB)"),
    source: Optional[str] = None,
    k: int = 10,
    model_name: str = "z-ai/glm-5.1",
    type_api: Literal["gemini", "huggingface", "openai", "qwen", "openrouter"] = "openrouter",
    save_result: bool = True,
):
    """
    Upload file PDF dan masukkan ke antrean analisis secara asinkron.
    Mengembalikan job_id yang bisa dicek melalui GET /analyze/{job_id}.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang diterima.")

    file_size = 0
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"Ukuran file melebihi batas maksimal ({MAX_FILE_SIZE / (1024*1024)}MB).")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan file: {str(e)}")

    doc_source = source or file.filename
    supabase_vdb = app_state["supabase_vdb"]
    chat_prompt = app_state["chat_prompt"]
    llm = get_llm(model_name=model_name, type_api=type_api)
    
    job_id = str(uuid.uuid4())

    background_tasks.add_task(
        process_analysis_task,
        job_id=job_id,
        tmp_path=tmp_path,
        tmp_dir=tmp_dir,
        doc_source=doc_source,
        supabase_vdb=supabase_vdb,
        llm=llm,
        chat_prompt=chat_prompt,
        k=k,
        save_result=save_result
    )

    return {
        "status": "accepted",
        "job_id": job_id,
        "message": "Dokumen sedang dianalisis di latar belakang. Gunakan job_id untuk mengecek status."
    }


@app.get("/analyze/{job_id}", tags=["Analisis"])
def get_analysis_status(job_id: str):
    """
    Mengecek status dan mengambil hasil dari analisis dokumen.
    """
    job = app_state["jobs"].get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID tidak ditemukan.")
    
    return job


@app.post("/seed", tags=["Knowledge Base"])
async def seed_knowledge(
    file: UploadFile = File(..., description="File PDF SDG Methodology (Max 10MB)"),
    source: Optional[str] = None,
    page_start: int = 0,
    page_end: Optional[int] = None,
):
    """
    Upload file PDF SDG Methodology dan tambahkan ke vector database Supabase.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang diterima.")

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"Ukuran file melebihi batas maksimal ({MAX_FILE_SIZE / (1024*1024)}MB).")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)

    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        doc_source = source or file.filename
        page_range = [page_start, page_end] if page_end else None
        supabase_vdb = app_state["supabase_vdb"]

        logger.info("Mulai seeding knowledge: %s", doc_source)

        added_ids = await run_in_threadpool(
            add_sdg_knowledge,
            path_file=tmp_path,
            source=doc_source,
            supabase_vdb=supabase_vdb,
            page_range=page_range,
        )

        logger.info("Seeding selesai: %s (%d chunks)", doc_source, len(added_ids))

        return {
            "status": "ok",
            "source": doc_source,
            "total_chunks_added": len(added_ids),
        }

    except Exception as e:
        logger.exception("Error saat seeding knowledge: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Terjadi error: {str(e)}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
