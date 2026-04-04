from app.database import inmemory_vdb_service, supabase_service
from app.infrastructure import embedding_service

from langchain_community.vectorstores import FAISS, SupabaseVectorStore
from langchain_core.embeddings import FakeEmbeddings
from supabase import Client

from dotenv import load_dotenv
import pytest
import os
import json

load_dotenv()
mock_path = "./tests/mocks/"


@pytest.mark.supabase
def test_supabase_init():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
    supabase = supabase_service.supabase_init(supabase_url=supabase_url, supabase_service_key=supabase_service_key)
    assert type(supabase) == Client


@pytest.mark.supabase
def test_supabase_vdb_init(supabase_client_init):
    supabase = supabase_client_init
    embeddings = FakeEmbeddings(size=1024)
    vdb = supabase_service.supabase_vdb_init(supabase=supabase, embeddings=embeddings)

    assert isinstance(vdb, SupabaseVectorStore)

    result = vdb.similarity_search_with_relevance_scores("test search", k=2)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0][1], float)


@pytest.mark.supabase
def test_fetch_sdg_indicator(supabase_client_init):
    data = supabase_service.fetch_sdg_indicator(supabase_client_init)
    assert len(data) > 100

    save_path = mock_path+"mock_SDG_indicator.json"
    if os.path.exists(save_path):
        pass
    else:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


@pytest.mark.integration(type_init="vdb")
def test_inmemory_vdb_init():
    embeddings = FakeEmbeddings(size=1024)
    vdb = inmemory_vdb_service.inmemory_vdb_init(embeddings=embeddings)
    assert isinstance(vdb, FAISS)

    example_data = {
        "texts":[
            "Dokumen penelitian tentang MBG",
            "Dokumen administrasi peraturan memasak MBG"
        ],
        "ids":['1', '2']
    }
    vdb.add_texts(**example_data)
    isi_vdb = vdb.docstore.search('1').page_content
    assert isinstance(isi_vdb, str)
    assert len(isi_vdb) > 5
