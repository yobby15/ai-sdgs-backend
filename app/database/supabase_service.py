import logging
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
import os

logger = logging.getLogger(__name__)

def supabase_init(supabase_url: str = None, supabase_service_key: str = None):
    if supabase_url is None:
        supabase_url = os.getenv("SUPABASE_URL")

    if supabase_service_key is None:
        supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
    supabase: Client = create_client(supabase_url, supabase_service_key)
    return supabase


def supabase_vdb_init(supabase:Client, embeddings):
    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="sdg_vectorstore",
        query_name="match_documents",
    )

    return vector_store


def fetch_sdg_indicator(supabase:Client):
    logger.debug("[START] : Fetch SDG Indicator from supabase")
    response = supabase.table("sdg_vectorstore").select("content, metadata, embedding").execute()
    sdg_data_rows = response.data
    logger.debug("[DONE] : Fetch SDG Indicator from supabase (length data:%s)", str(len(sdg_data_rows)))
    return sdg_data_rows

