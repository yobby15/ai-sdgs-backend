from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from typing import Literal
import logging

logger = logging.getLogger(__name__)

def embedding_init(
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        type_run: Literal["local", "google_genai", "openai"] = "local",
):  
    logger.debug("Start Embedding initiation (input parameter: %s) ", {
        "model_name":model_name,
        "type_run":type_run,
    })
    try:
        match type_run:
            case "local":
                model_kwargs = {'device': 'cpu'}
                encode_kwargs = {'normalize_embeddings': True}

                embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
            case "google_genai":
                embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
            case "openai":
                embeddings = OpenAIEmbeddings(model=model_name)
    except Exception as e:
        logger.exception("Terjadi Error saat inisiasi embedding: %s", str(e))
        raise        

    logger.debug("Done Embedding initiation")
    return embeddings

