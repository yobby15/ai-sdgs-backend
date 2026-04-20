from app.core import input_doc, retrieval
from app.utils import text_processing
from app.infrastructure import embedding_service
from app.database import inmemory_vdb_service

import pytest
import json
import os
import numpy as np
import numbers

mock_path = "./tests/mocks/"

@pytest.mark.unit
def test_convert_to_md():
    example_path = "./data/Sample Documents/TOR-PM.pdf"
    result_md = input_doc.convert_to_md(example_path)
    assert isinstance(result_md, str)
    assert len(result_md) > 100


@pytest.mark.unit
def test_split_and_clean_pages(generate_md):
    splitted_pages = text_processing.split_and_clean_pages(generate_md)
    assert isinstance(splitted_pages, list)
    assert len(splitted_pages) > 1


@pytest.mark.unit
def test_chunk_text(generate_md):
    chunked_text = text_processing.chunk_text(generate_md)
    assert len(chunked_text[0]) > 100


# @pytest.mark.unit
# def test_pages_to_json_format(generate_chunked_pages):
#     json_pages = text_processing.pages_to_json_format(generate_chunked_pages, type_doc="sdg_evidence", source="TOR-PM.pdf")
#     assert isinstance(json_pages, list)
#     assert isinstance(json_pages[0], dict)
#     assert len(json_pages) > 3

#     save_path = mock_path+"mock_extracted_TOR-PM.json"
#     if os.path.exists(save_path):
#         pass
#     else:
#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(json_pages, f, ensure_ascii=False, indent=4)

#     assert os.path.exists(save_path)


# @pytest.mark.unit
# def test_matching_SDG(generate_SDG_indicator, load_inmemory_vdb):
#     assert load_inmemory_vdb.index.ntotal > 5
#     assert len(generate_SDG_indicator) > 100
#     matched_sdg = retrieval.matching_SDG(generate_SDG_indicator, load_inmemory_vdb)
#     assert isinstance(matched_sdg, list)
#     assert len(matched_sdg) > 10
#     assert isinstance(matched_sdg[0]["score"], numbers.Real)

#     save_path = mock_path+"mock_matched_sdg.json"
#     if os.path.exists(save_path):
#         pass
#     else:
#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(matched_sdg, f, ensure_ascii=False, indent=4)
    
#     assert os.path.exists(save_path)


@pytest.mark.unit
def test_build_graph_prompt(generate_matched_sdg, load_inmemory_vdb):
    matched_sdg = generate_matched_sdg
    retrieval_result = retrieval.build_graph_prompt(matched_sdg[:10], load_inmemory_vdb, window_size=2)

    assert isinstance(retrieval_result, str)
    assert len(retrieval_result) > 100

    save_path = mock_path+"retrieval_result.txt"
    if os.path.exists(save_path):
        pass
    else:
        with open(save_path, "w") as f:
            f.write(retrieval_result)
    
    assert os.path.exists(save_path)


@pytest.mark.unit
def test_detect_metadata(generate_splitted_pages: list[str]):
    splitted_pages = generate_splitted_pages
    result = input_doc.detect_metadata(splitted_pages)
    assert isinstance(result, dict)


@pytest.mark.unit(type_test="priority")
def test_extract_metadata(generate_md):
    md_text = generate_md
    metadata = input_doc.extract_metadata(md_text, "test_article")
    assert isinstance(metadata, dict)
    assert isinstance(metadata['abstract'], str)
    assert isinstance(metadata['title'], str)
    assert isinstance(metadata['conclusion'], str)
