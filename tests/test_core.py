from app.core import input_doc, retrieval

import pytest
import os
from dotenv import load_dotenv


load_dotenv()
mock_path = "./tests/mocks/"


@pytest.mark.core(type_test="input")
def test_input_document():
    filename = "ACHMAD KAUTSAR_1.pdf"
    metadata = input_doc.input_document(path_file=mock_path+filename, source=filename)

    assert isinstance(metadata, dict)
    assert isinstance(metadata['abstract'], str)
    assert isinstance(metadata['title'], str)
    assert isinstance(metadata['conclusion'], str)


@pytest.mark.core(type_test="retrieval")
def test_retrieval_SDG(load_supabase_vdb, generate_metadata):
    vdb = load_supabase_vdb
    metadata = generate_metadata
    retrieval_result = retrieval.retrival_SDG(metadata_article=metadata, vdb=vdb, k=5)

    assert isinstance(retrieval_result, str)
    assert len(retrieval_result) > 500

    save_path = mock_path+"mock_retrieval_result.txt"
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            f.write(retrieval_result)
    else:
        pass

    assert os.path.exists(save_path)