from app.core import input_doc, retrieval

import pytest
import os
from dotenv import load_dotenv


load_dotenv()
mock_path = "./tests/mocks/"


@pytest.mark.core(type_test="input")
def test_input_document_journal():
    filename = "ACHMAD KAUTSAR_1.pdf"
    metadata = input_doc.input_document(path_file=mock_path+filename, source=filename)

    assert isinstance(metadata, dict)
    assert isinstance(metadata['abstract'], str)
    assert isinstance(metadata['title'], str)
    assert isinstance(metadata['conclusion'], str)

@pytest.mark.core(type_test="input")
def test_input_document_administration():
    filename = "TOR-PM.pdf"
    result_extraction = input_doc.input_document(path_file=mock_path+filename, source=filename)

    assert isinstance(result_extraction, list)
    assert isinstance(result_extraction[0], input_doc.Document)

@pytest.mark.core(type_test="input")
def test_input_document_sdg_knowledge():
    filename = "Sustainability Impact Ratings Methodology 2026.pdf"
    result_extraction = input_doc.input_document(path_file=mock_path+filename, source=filename, type_doc="sdg_knowledge", page_range=[10, 153])

    assert isinstance(result_extraction, list)
    assert isinstance(result_extraction[0], input_doc.Document)
    assert len(result_extraction) > 100
        

@pytest.mark.core(type_test="retrieval")
def test_metadata_retrieval_SDG(load_supabase_vdb, generate_metadata):
    vdb = load_supabase_vdb
    metadata = generate_metadata
    retrieval_result = retrieval.metadata_retrival_SDG(metadata_article=metadata, vdb=vdb, k=5)

    assert isinstance(retrieval_result, str)
    assert len(retrieval_result) > 500

    save_path = mock_path+"mock_metadata_retrieval_result.txt"
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            f.write(retrieval_result)
    else:
        pass

    assert os.path.exists(save_path)


@pytest.mark.core(type_test="retrieval-1")
def test_chunks_retrieval_SDG(load_supabase_vdb, generate_extracted_data):
    vdb = load_supabase_vdb
    extracted_data: list[dict] = generate_extracted_data
    chunks_document: list[input_doc.Document] = [input_doc.Document(page_content=chunk["text"], metadata=chunk["metadata"]) for chunk in extracted_data]
    retrieval_result = retrieval.chunks_retrieval_SDG(documents=chunks_document, vdb=vdb, k=2)

    assert isinstance(retrieval_result, str)
    assert len(retrieval_result) > 500

    save_path = mock_path+"mock_chunks_retrieval_result.txt"
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            f.write(retrieval_result)
    else:
        pass

    assert os.path.exists(save_path)