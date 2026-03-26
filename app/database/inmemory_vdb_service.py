from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
import faiss

import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

def inmemory_vdb_init(embeddings, vector_length:int = 1024):
    index = faiss.IndexFlatIP(vector_length)
    inmemory_vdb = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    return inmemory_vdb


def vdb_embedding(text_input:str | list[str], embeddings):

    logger.info("Embedding Input Document")
    logger.debug("[START] : Embedding Input Document (input param:%s)", {
        "text_input_type":type(text_input),
        "text_input_length":len(text_input),
        "embeddings_type":type(embeddings)
    })
    
    start_time = time.time()
    if type(text_input) == str:
        embedded_docs = [embeddings.embed_query(text_input)]
        embedded_docs = np.array(embedded_docs).astype('float32')
        faiss.normalize_L2(embedded_docs)
        time_execution = time.time() - start_time
        logger.debug("[END] : Embedding Input Document (time execution:%s)", str(time_execution))
        return embedded_docs.tolist()[0]

    elif type(text_input) == list:
        embedded_docs = embeddings.embed_documents(text_input)
        embedded_docs = np.array(embedded_docs).astype('float32')
        faiss.normalize_L2(embedded_docs)
        time_execution = time.time() - start_time
        logger.debug("[END] : Embedding Input Document (time execution:%s)", str(time_execution))
        return embedded_docs.tolist()
    

