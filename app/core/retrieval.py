from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore

import faiss
import json
import numpy as np
import logging
from typing import Literal

logger = logging.getLogger(__name__)


def matching_SDG(sdg_data_rows: list, inmemory_vdb, k:int = 10):
    logger.info("Matching SDG Indicator")
    all_matches = []

    for row in sdg_data_rows:
        sdg_vector = [json.loads(row["embedding"]) if not isinstance(row["embedding"], list) else row["embedding"]]
        sdg_vector = np.array(sdg_vector).astype('float32')
        faiss.normalize_L2(sdg_vector)
        sdg_vector = sdg_vector.tolist()[0]
        sdg_text = row["content"]
        sdg_metadata = row["metadata"]
        
        # Cari k chunk tercocok untuk indikator
        matched_docs = inmemory_vdb.similarity_search_with_score_by_vector(
            embedding=sdg_vector, 
            k=k
        )
        
        for doc, score in matched_docs:
            all_matches.append({
                "score": float(score),
                "sdg_content": sdg_text,
                "sdg_metadata": sdg_metadata,
                "admin_content": doc.page_content,
                "admin_metadata": doc.metadata
            })

    sorted_match = sorted(all_matches, key=lambda x: x["score"], reverse=True)
    return sorted_match



def build_graph_prompt(
        sdg_matches:list,
        type_input_retrieval: Literal["metadata", "chunk"] = "metadata"
        # inmemory_vdb,
        # window_size:int=1
    ):
    """
    `window_size`: berapa banyak chunk tambahan di depan dan belakang yang ingin diambil (default 1)
    """
    logger.info("Build Graph Prompt")

    unique_admin_ids = set()
    unique_sdg = {}
    mapping = []
    admin_content = dict()
    # admin_ids = list(inmemory_vdb.docstore._dict.keys())

    for m in sdg_matches:
        match type_input_retrieval:
            case "metadata":
                a_id = m['admin_metadata']['type_metadata']
                unique_admin_ids.add(str(a_id))
            case "chunk":
                a_id = m['admin_metadata']['global_chunk_id']
                unique_admin_ids.add(int(a_id))
                
        s_id = m['sdg_metadata']['global_chunk_id']
        
        # for offset in range(-window_size, window_size + 1):
        #     unique_admin_ids.add(str(a_id + offset))
        admin_content[str(a_id)] = m['admin_content']
            
        unique_sdg[s_id] = m['sdg_content']

        mapping.append({
            "input_doc_chunk_id": a_id,
            "SDG_reference_chunk_id": s_id,
            "similarity_score": float(m['score'])
        })

    retrieval_result = "MAP HUBUNGAN:\n" + json.dumps(mapping, indent=2) + "\n\n"
    
    retrieval_result += "ISI CHUNK (DOKUMEN INPUT):\n"
    for chunk_id in sorted(unique_admin_ids):
        teks_chunk = admin_content[str(chunk_id)]
        retrieval_result += f"[{chunk_id}]: {teks_chunk}\n\n\n"
    
    retrieval_result += "\nISI CHUNK (SDG REFERENCES):\n"
    for k, v in unique_sdg.items():
        retrieval_result += f"[{k}]: {v}\n\n\n"

    return retrieval_result


def metadata_retrival_SDG(metadata_article: dict, vdb: SupabaseVectorStore, k=10):
    logger.info("Retrieval: Matching SDG Indicator from metadata document")
    all_matches = []

    for i, (key, value) in enumerate(metadata_article.items()):
        if key == "document_name":
            continue

        query = value
        type_metadata = key
        matches_docs = vdb.similarity_search_with_relevance_scores(query, k=k)

        for doc, score in matches_docs:
            all_matches.append({
                "score": float(score),
                "sdg_content": doc.page_content,
                "sdg_metadata": doc.metadata,
                "admin_content": query,
                "admin_metadata": {
                    "type_metadata":type_metadata,
                    "global_chunk_id":i,
                }
            })

    sorted_match = sorted(all_matches, key=lambda x: x["score"], reverse=True)
    retrieval_result = build_graph_prompt(sorted_match, type_input_retrieval="metadata")
    
    return retrieval_result


def chunks_retrieval_SDG(documents: list[Document], vdb: SupabaseVectorStore, k=10):
    logger.info("Retrieval: Matching SDG Indicator from chunks of document")
    all_matches = []

    for chunk_doc in documents:
        query = chunk_doc.page_content
        metadata = chunk_doc.metadata
        matches_docs = vdb.similarity_search_with_relevance_scores(query=query, k=k)

        for doc, score in matches_docs:
            all_matches.append({
                "score": float(score),
                "sdg_content": doc.page_content,
                "sdg_metadata": doc.metadata,
                "admin_content": query,
                "admin_metadata": metadata
            })

    sorted_match = sorted(all_matches, key=lambda x: x["score"], reverse=True)
    retrieval_result = build_graph_prompt(sorted_match, type_input_retrieval="chunk")
    
    return retrieval_result