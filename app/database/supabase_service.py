import logging
from supabase.client import Client, create_client

logger = logging.getLogger(__name__)

def supabase_init(supabase_url: str, supabase_service_key: str):
    supabase: Client = create_client(supabase_url, supabase_service_key)
    return supabase


def fetch_sdg_indicator(supabase:Client):
    logger.debug("[START] : Fetch SDG Indicator from supabase")
    response = supabase.table("sdg_vectorstore").select("content, metadata, embedding").execute()
    sdg_data_rows = response.data
    logger.debug("[DONE] : Fetch SDG Indicator from supabase (length data:%s)", str(len(sdg_data_rows)))
    return sdg_data_rows
