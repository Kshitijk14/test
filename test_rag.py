import pytest
from unittest.mock import patch
import os

from llama_index.core.llms import MockLLM as BaseMockLLM, ChatMessage, MessageRole
from rag import build_rag, Settings
from utils.config import CONFIG

# Load query from your config
QUERY = CONFIG["QUERY"]
HF_MODEL = "Qwen/Qwen3-8B"

# ✅ Create a custom mock LLM with a defined `chat` method
class CustomMockLLM(BaseMockLLM):
    def chat(self, messages, **kwargs):
        return ChatMessage(content="Mocked response from Hugging Face LLM", role=MessageRole.ASSISTANT)

@pytest.fixture
def mock_hf_llm():
    """Fixture to mock the Hugging Face LLM using CustomMockLLM."""
    mock_llm = CustomMockLLM()

    # Set IS_TESTING to ensure resolve_llm behaves correctly
    os.environ["IS_TESTING"] = "1"

    # ✅ Patch the get_llm_model_hf function to return the mock
    with patch('utils.rag.get_models.get_llm_model_hf', return_value=mock_llm):
        # Set the mocked LLM in Settings
        Settings._llm = mock_llm
        yield mock_llm

    # ✅ Clean up after test
    del os.environ["IS_TESTING"]
    if hasattr(Settings, '_llm'):
        del Settings._llm

@pytest.mark.parametrize("question", [QUERY, None], ids=["explicit_query", "default_query"])
def test_build_rag_with_hf_llm(mock_hf_llm, question):
    """Test build_rag with a mocked Hugging Face LLM."""
    response = build_rag(question=question, llm=mock_hf_llm)
    # ✅ Base assertions
    assert response is not None, "Response should not be None"
    assert isinstance(response, str) or hasattr(response, 'response'), "Response should be a string or have a response attribute"
    # ✅ Content assertions
    if isinstance(response, str):
        assert len(response.strip()) > 0, "Response should not be empty"
        print("\nLLM Response (str):", response)  # Print if it's a plain string
    elif hasattr(response, 'response'):
        assert len(response.response.strip()) > 0, "Response content should not be empty"
        print("\nLLM Response:", response.response)  # Print the response attribute
    # ✅ Optional: check for source chunks
    if hasattr(response, 'source_nodes'):
        assert len(response.source_nodes) > 0, "Should retrieve at least one chunk"