from app.core import input_doc, retrieval
from app.database import supabase_service, vdb_utils
from app.infrastructure import embedding_service, llm_agent_service, prompt_agent
from app.utils import text_processing

from langchain_community.vectorstores import FAISS, SupabaseVectorStore
from supabase import Client

from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import os
import uuid
import time
import json
import logging
from typing import Literal

load_dotenv()
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"), # Simpan ke file
        logging.StreamHandler()         # Munculkan di terminal
    ],
    encoding="utf-8"
)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("hpack").setLevel(logging.INFO)


logger = logging.getLogger(__name__)

def analyze_document(
        path_file:str,
        save_path:str,
        source:str,
        # supabase:Client,
        # embeddings,
        # inmemory_vdb : FAISS,
        supabase_vdb : SupabaseVectorStore,
        llm,
        chat_prompt,
        k:int = 2,
        # window_size:int = 2
) -> dict:
        logger.debug("[START] : Analyzing document")
        model_name = llm.model_dump()['name']
        id_request = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        start_time = time.time()

        # chunk_ids, texts, metadatas = input_doc.input_document(path_file=path_file, source=source)
        # embedded_docs = inmemory_vdb_service.vdb_embedding(
        #         text_input=texts,
        #         embeddings=embeddings
        # )

        # inmemory_vdb.add_embeddings(
        #         text_embeddings=[(text, emb) for text, emb in zip(texts, embedded_docs)],
        #         metadatas=metadatas,
        #         ids=chunk_ids
        # )

        # sdg_data_rows = supabase_service.fetch_sdg_indicator(supabase=supabase)

        # matches_chunk = retrieval.matching_SDG(
        #         sdg_data_rows=sdg_data_rows,
        #         inmemory_vdb=inmemory_vdb
        # )

        # retrieval_result = retrieval.build_graph_prompt(matches_chunk[:k], inmemory_vdb=inmemory_vdb, window_size=window_size)

        extraction_result = input_doc.input_document(path_file=path_file, source=source)
        if isinstance(extraction_result, dict):
            retrieval_result = retrieval.metadata_retrival_SDG(metadata_article=extraction_result, vdb=supabase_vdb, k=k)
        elif isinstance(extraction_result, list):
            retrieval_result = retrieval.chunks_retrieval_SDG(documents=extraction_result, vdb=supabase_vdb, k=k)



        prompt_value = chat_prompt.format_prompt(text=retrieval_result)
        full_prompt = prompt_value.to_string()

        logger.info("Send prompt to LLM")
        try:
            logger.debug("[START] : LLM Agent Analyzing (total chars in prompt:%s)", len(full_prompt))
            llm_output = llm.invoke(prompt_value)
            logger.debug("[END] : LLM Agent Analyzing")
        except Exception as e:
            logger.error("Terjadi error pada LLM\n\n%s", e)
            raise

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
        
        logger.debug("[END] : Analyzing document")
        return final_result



def add_sdg_knowledge(
        path_file: str,
        source: str,
        supabase_vdb: SupabaseVectorStore,
        page_range:list[int, int] = None,
        special_page:list[int] = []
):
    logger.debug("[START] : Adding SDG Knowledge (input_param: %s)")
    start_time = time.time()
    extraction_result: list[input_doc.Document] = input_doc.input_document(
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
    logger.debug("[END] : Adding SDG Knowledge")
    
    return added_data_ids
    

def main(
        path_file: str,
        save_path: str = None,
        type_run: Literal["analyze_document", "add_sdg_knowledge"] = "analyze_document",
        source: str = None,
        k: int = 10,
        window_size: int = 2,
        page_range:list[int, int] = None,
        special_page:list[int] = []
):
    match type_run:
        case "analyze_document":
            logger.info("Proses Analisis")
        case "add_sdg_knowledge":
            logger.info("SDG Adding Knowledge to Vector DB")

    logger.debug("Parameter Input: %s",
    {"path_file": path_file,
     "type_run":type_run,
    "save_path":save_path,
    "source":source,
    "k":k,
    "window_size":window_size,
    })

    # input validation
    if not os.path.exists(path_file):
        logger.error("Input File (%s) not found", path_file)
        return f"Input File ({path_file}) not found"
    if save_path:
        if not os.path.exists(save_path):
            logger.error("Save Path Folder (%s) not found", save_path)
            return f"Save Path Folder ({save_path}) not found"
    
    source = os.path.basename(path_file) if source == None else source

    # Initiation
    logger.info("Initiation Process")
    logger.debug("[START] : Initiation Process")
    embeddings = embedding_service.embedding_init()
    supabase = supabase_service.supabase_init(supabase_url=os.getenv("SUPABASE_URL"), supabase_service_key=os.getenv("SUPABASE_SERVICE_KEY"))
    supabase_vdb = supabase_service.supabase_vdb_init(supabase=supabase, embeddings=embeddings)
    # inmemory_vdb = inmemory_vdb_service.inmemory_vdb_init(embeddings=embeddings, vector_length=1024)
    if type_run == "analyze_document":
        llm_agent = llm_agent_service.model_init(model_name="Qwen/Qwen3-Coder-Next", type_model="huggingface")
        chat_prompt = prompt_agent.FULL_CHAT_PROMPT
    logger.debug("[END] : Initiation Process")

    match type_run:
        case "analyze_document":
            logger.info("Anlyzing Document")
            result = analyze_document(
                path_file=path_file,
                save_path=save_path,
                source=source,
                #  supabase=supabase,
                #  embeddings=embeddings,
                #  inmemory_vdb=inmemory_vdb,
                supabase_vdb=supabase_vdb,
                llm=llm_agent,
                chat_prompt=chat_prompt,
                k=k,
                #  window_size=window_size
            )
        case "add_sdg_knowledge":
            added_data_ids = add_sdg_knowledge(
                path_file=path_file,
                source=source,
                supabase_vdb=supabase_vdb,
                page_range=page_range,
                special_page=special_page
            )


if __name__ == "__main__":
    main(
        path_file="./data/SDGs_knowledge_dataset/Sustainability Impact Ratings Methodology 2026.pdf",
        type_run="add_sdg_knowledge",
        page_range=[10,153],
        special_page=[10, 19, 30, 36, 42, 54, 63, 71, 80, 86, 97, 106, 114, 122, 131, 138, 146]
    )

    # main(
    #     path_file="./data/Sample Documents/jurnal-unesa/ACHMAD KAUTSAR_4.pdf",
    #     save_path="./ai_result/"
    # )