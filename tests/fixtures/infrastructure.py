from app.infrastructure import embedding_service
import pytest


@pytest.fixture
def load_default_embedding():
    embeddings = embedding_service.embedding_init(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

@pytest.fixture
def load_mini_embedding():
    embeddings = embedding_service.embedding_init(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings