from app.database import supabase_service, inmemory_vdb_service
from app.infrastructure import embedding_service

from supabase import Client

import os
import pytest

mock_path = "./tests/mocks"

@pytest.fixture
def supabase_client_init() -> Client:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
    supabase = supabase_service.supabase_init(supabase_url=supabase_url, supabase_service_key=supabase_service_key)
    return supabase

@pytest.fixture
def load_supabase_vdb(supabase_client_init):
    supabase = supabase_client_init
    embeddings = embedding_service.embedding_init()
    vdb = supabase_service.supabase_vdb_init(supabase=supabase, embeddings=embeddings)
    return vdb

@pytest.fixture
def load_inmemory_vdb(load_mini_embedding, generate_extracted_data):
    embeddings = load_mini_embedding
    vdb = inmemory_vdb_service.inmemory_vdb_init(embeddings=embeddings, vector_length=384)
    save_path = mock_path+"mock_faiss_index"
    if os.path.exists(save_path):
        vdb = vdb.load_local(save_path,
                       embeddings=embeddings,
                       allow_dangerous_deserialization=True)
    else:
        data_sample = generate_extracted_data
        texts = [item["text"] for item in data_sample]
        metadatas = [item["metadata"] for item in data_sample]
        ids = [str(i) for i in range(len(data_sample))]
        vectors = inmemory_vdb_service.vdb_embedding(text_input=texts, embeddings=embeddings)
        vdb.add_embeddings(
            text_embeddings=[(txt, embed) for txt, embed in zip(texts, vectors)],
            metadatas=metadatas,
            ids=ids
        )
        vdb.save_local(save_path)
    return vdb