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

@pytest.fixture
def generate_md():
    example_md = input_doc.convert_to_md("./data/Sample Documents/TOR-PM.pdf")
    return example_md

@pytest.fixture
def generate_chunked_pages(generate_md):
    splitted_pages = text_processing.split_and_clean_pages(generate_md)
    chunked_pages = [text_processing.chunk_text(text) for text in splitted_pages]
    return chunked_pages

@pytest.fixture
def generate_extracted_data():
    file_path = mock_path+"mock_extracted_TOR-PM.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@pytest.fixture
def generate_SDG_indicator(load_mini_embedding):
    file_path = mock_path+"mock_SDG_indicator.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

@pytest.fixture
def generate_matched_sdg():
    file_path = mock_path+"mock_matched_sdg.json"
    with open(file_path, "r", encoding="utf-8") as f:
        matched_sdg = json.load(f)
    return matched_sdg

@pytest.fixture
def load_mini_embedding():
    embeddings = embedding_service.embedding_init(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

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


@pytest.mark.unit
def test_pages_to_json_format(generate_chunked_pages):
    json_pages = text_processing.pages_to_json_format(generate_chunked_pages, type_doc="sdg_evidence", source="TOR-PM.pdf")
    assert isinstance(json_pages, list)
    assert isinstance(json_pages[0], dict)
    assert len(json_pages) > 3

    save_path = mock_path+"mock_extracted_TOR-PM.json"
    if os.path.exists(save_path):
        pass
    else:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(json_pages, f, ensure_ascii=False, indent=4)

    assert os.path.exists(save_path)


@pytest.mark.unit
def test_matching_SDG(generate_SDG_indicator, load_inmemory_vdb):
    assert load_inmemory_vdb.index.ntotal > 5
    assert len(generate_SDG_indicator) > 100
    matched_sdg = retrieval.matching_SDG(generate_SDG_indicator, load_inmemory_vdb)
    assert isinstance(matched_sdg, list)
    assert len(matched_sdg) > 10
    assert isinstance(matched_sdg[0]["score"], numbers.Real)

    save_path = mock_path+"mock_matched_sdg.json"
    if os.path.exists(save_path):
        pass
    else:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(matched_sdg, f, ensure_ascii=False, indent=4)
    
    assert os.path.exists(save_path)


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