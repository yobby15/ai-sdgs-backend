from app.core import input_doc, retrieval
from app.database import supabase_service, inmemory_vdb_service
from app.infrastructure import embedding_service, llm_agent_service, prompt_agent
from app.utils import text_processing

from langchain_community.vectorstores import FAISS
from supabase import Client

from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import os
import uuid
import time
import json
import logging

load_dotenv()
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"), # Simpan ke file
        logging.StreamHandler()         # Munculkan di terminal
    ]
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("hpack").setLevel(logging.INFO)


logger = logging.getLogger(__name__)

def analyze_document(
        path_file:str,
        save_path:str,
        source:str,
        supabase:Client,
        embeddings,
        inmemory_vdb : FAISS,
        llm,
        chat_prompt,
        k:int = 5,
        window_size:int = 2
) -> dict:
        model_name = llm.model_dump()['name']
        id_request = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        start_time = time.time()

        chunk_ids, texts, metadatas = input_doc.input_document(path_file=path_file, source=source)
        embedded_docs = inmemory_vdb_service.vdb_embedding(
                text_input=texts,
                embeddings=embeddings
        )

        inmemory_vdb.add_embeddings(
                text_embeddings=[(text, emb) for text, emb in zip(texts, embedded_docs)],
                metadatas=metadatas,
                ids=chunk_ids
        )

        sdg_data_rows = supabase_service.fetch_sdg_indicator(supabase=supabase)

        matches_chunk = retrieval.matching_SDG(
                sdg_data_rows=sdg_data_rows,
                inmemory_vdb=inmemory_vdb
        )

        retrieval_result = retrieval.build_graph_prompt(matches_chunk[:k], inmemory_vdb=inmemory_vdb, window_size=window_size)

        prompt_value = chat_prompt.format_prompt(text=retrieval_result)
        full_prompt = prompt_value.to_string()

        llm_output = llm.invoke(prompt_value)
        final_result = text_processing.repair_llm_json(llm_output.content)

        time_execution = time.time() - start_time
        json_result = {
                "id_request": id_request,
                "timestamp": timestamp,
                "model_name": model_name,
                "time_execution": time_execution,
                "input_model": full_prompt,
                "output_model": llm_output.model_dump(),
                "result":final_result
        }


        folder = Path(save_path)
        filename = f"result_{source[:-4]}_{id_request}.json"

        file_path = folder / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, indent=4, ensure_ascii=False)
        
        return final_result


def main(
        path_file: str,
        save_path: str,
        source: str = None,
        k: int = 10,
        window_size: int = 2
):
    logger.info("Proses Analisis")
    logger.debug("Parameter Input: %s",
    {"path_file": path_file,
    "save_path":save_path,
    "source":source,
    "k":k,
    "window_size":window_size,
    })

    # input validation
    if not os.path.exists(path_file):
        logger.error("Input File (%s) not found", path_file)
        return f"Input File ({path_file}) not found"
    if not os.path.exists(save_path):
        logger.error("Save Path Folder (%s) not found", save_path)
        return f"Save Path Folder ({save_path}) not found"
    
    source = os.path.basename(path_file) if source == None else source

    # Initiation
    logger.info("START Initiation Process")
    embeddings = embedding_service.embedding_init()
    llm_agent = llm_agent_service.model_init(model_name="Qwen/Qwen3-Coder-Next", type_model="huggingface")
    supabase = supabase_service.supabase_init(supabase_url=os.getenv("SUPABASE_URL"), supabase_service_key=os.getenv("SUPABASE_SERVICE_KEY"))
    inmemory_vdb = inmemory_vdb_service.inmemory_vdb_init(embeddings=embeddings, vector_length=1024)
    chat_prompt = prompt_agent.FULL_CHAT_PROMPT
    logger.info("DONE Initiation Process")

    logger.info("START Anlyzing Document")
    result = analyze_document(
         path_file=path_file,
         save_path=save_path,
         source=source,
         supabase=supabase,
         embeddings=embeddings,
         inmemory_vdb=inmemory_vdb,
         llm=llm_agent,
         chat_prompt=chat_prompt,
         k=k,
         window_size=window_size
    )
    logger.info("DONE Anlyzing Document")
    logger.debug("Analyse Output: %s", str(result))


if __name__ == "__main__":
    main(
         path_file="./data/Sample Documents/ADI PRANOTO_1.pdf",
         save_path="./ai_result",
    )

