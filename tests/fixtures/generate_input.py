from app.utils import text_processing
from app.core import input_doc

import pytest
import os
import json

mock_path = "./tests/mocks/"

@pytest.fixture
def generate_md() -> str:
    with open(mock_path+"ACHMAD KAUTSAR_1.md", "r", encoding="utf-8") as f:
        example_md = f.read()
    return example_md

@pytest.fixture
def generate_splitted_pages(generate_md) -> list[str]:
    splitted_pages = text_processing.split_and_clean_pages(generate_md)
    return splitted_pages

@pytest.fixture
def generate_chunked_pages(generate_splitted_pages):
    splitted_pages = generate_splitted_pages
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
def generate_metadata_indices(generate_splitted_pages: list[str]):
    splitted_pages = generate_splitted_pages
    metadata_indices = input_doc.detect_metadata(splitted_pages)
    return metadata_indices

@pytest.fixture
def generate_metadata(generate_md):
    md_text = generate_md
    metadata = input_doc.extract_metadata(md_text, "test_article")
    return metadata