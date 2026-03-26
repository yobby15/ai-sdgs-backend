from app.infrastructure import embedding_service
from app.infrastructure import llm_agent_service

from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qwq import ChatQwen

from langchain_core.messages import AIMessage

import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.integration(type_init="embedding")
def test_embedding_init_default():
    embeddings = embedding_service.embedding_init()
    assert isinstance(embeddings, HuggingFaceEmbeddings)
    
    embed_result = embeddings.embed_query("Hello World!")
    assert isinstance(embed_result, list)
    assert len(embed_result) == 1024


@pytest.mark.integration(type_init="embedding")
def test_embedding_init_gemini():
    embeddings = embedding_service.embedding_init(model_name="gemini-embedding-001", type_run='google_genai')
    assert isinstance(embeddings, GoogleGenerativeAIEmbeddings)
    
    embed_result = embeddings.embed_query("Hello World!")
    assert isinstance(embed_result, list)


@pytest.mark.integration(type_init="embedding")
def test_embedding_init_openai():
    embeddings = embedding_service.embedding_init(model_name="text-embedding-3-small", type_run="openai")
    assert isinstance(embeddings, OpenAIEmbeddings)
    
    embed_result = embeddings.embed_query("Hello World!")
    assert isinstance(embed_result, list)


@pytest.mark.integration(type_init="default-llm")
def test_llm_init_default():
    llm = llm_agent_service.model_init()
    assert isinstance(llm, ChatGoogleGenerativeAI)

    result = llm.invoke("Hello gemini")
    assert isinstance(result, AIMessage)


@pytest.mark.integration(type_init="llm")
def test_llm_init_qwen():
    llm = llm_agent_service.model_init(model_name="Qwen/Qwen3-Coder-Next", type_model="huggingface")
    assert isinstance(llm, ChatHuggingFace)

    result = llm.invoke("Hello Qwen")
    assert isinstance(result, AIMessage)