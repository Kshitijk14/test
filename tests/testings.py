import pytest
from unittest.mock import MagicMock
from utils.rag.populate import get_chunk_id
from utils.rag.get_models import get_embedding_model, get_llm_model
from utils.rag.get_prompt import RESPONSE_SYNTHESIS_PROMPT
from unittest.mock import patch
from rag import build_rag
import os

def test_get_chunk_id():
    text = "HELLO THIS IS THE TEST TEXT FOR THE PYTEST FOR THE TEST I AM GIVING FOR THE REPO TEST"
    file_name = "test_file"
    chunk_idx = 1
    result = get_chunk_id(text, file_name, chunk_idx)
    assert isinstance(result, str)
    assert file_name in result
    assert f"chunk{chunk_idx}" in result

# get_embedding_model
@pytest.fixture
def mock_huggingface_embedding():
    return MagicMock()


def test_get_embedding_model():
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    with patch('utils.rag.get_models.HuggingFaceEmbedding') as MockEmbedding:
        mock_instance = MockEmbedding.return_value
        mock_instance.model_name = embedding_model
        result = get_embedding_model(embedding_model)
        assert result.model_name == embedding_model


# get_llm_model
@pytest.fixture
def mock_openai():
    return MagicMock()



def test_get_llm_model():
    llm_model = "gpt-3.5-turbo"
    
    with patch('utils.rag.get_models.OpenAI') as MockOpenAI:
        mock_instance = MockOpenAI.return_value
        mock_instance.model = llm_model
        
        result = get_llm_model(llm_model)
        
        assert result.model == llm_model


# RESPONSE_SYNTHESIS_PROMPT
def test_prompt_template():
    assert "Answer in 2-3 sentences only." in RESPONSE_SYNTHESIS_PROMPT.template

# build_rag function
@pytest.fixture
def mock_build_rag():
    os.makedirs = MagicMock()
    return build_rag

def test_build_rag(mock_build_rag):

    build_rag()  
    assert os.makedirs.called

