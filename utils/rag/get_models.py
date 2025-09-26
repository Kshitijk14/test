import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
try:
    from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI  # type: ignore
except Exception:  # ImportError or module missing
    HuggingFaceInferenceAPI = None  # type: ignore
# from llama_index.llms import HuggingFaceInferenceAPI
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
try:
    from llama_index.llms.groq import Groq  # type: ignore
except Exception:  # ImportError or module missing
    Groq = None  # type: ignore
from huggingface_hub import InferenceClient

from ..config import CONFIG
from dotenv import load_dotenv

load_dotenv()

TEMPERATURE = CONFIG["TEMPERATURE"]
TIMEOUT = CONFIG["TIMEOUT"]
LLM = CONFIG["LLM_MODEL"]

def get_embedding_model(embedding_model: str):
    embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    return embed_model

def _resolve_hf_token() -> str:
    """Resolve HF token from multiple sources with precedence.

    Precedence: HF_TOKEN env > HUGGINGFACEHUB_API_TOKEN env > params.yaml:HUGGINGFACE_API_KEY
    Also synchronizes env vars so both HF_TOKEN and HUGGINGFACEHUB_API_TOKEN are set.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or CONFIG.get("HUGGINGFACE_API_KEY")
    if not token:
        raise ValueError(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN, or configure HUGGINGFACE_API_KEY in params.yaml."
        )
    # Keep envs in sync for downstream libs/CLIs
    os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)
    return token


def get_llm_model_hf(llm_model: str = LLM, temp: float=TEMPERATURE, timeout: int=TIMEOUT):
    """Return a Hugging Face Inference API-backed LLM for LlamaIndex.

    Uses token from HF_TOKEN/HUGGINGFACEHUB_API_TOKEN/params.yaml and keeps envs in sync.
    """
    if HuggingFaceInferenceAPI is None:
        raise ImportError(
            "Hugging Face LLM backend is not available. Install 'llama-index-llms-huggingface-api'."
        )
    token = _resolve_hf_token()
    llm = HuggingFaceInferenceAPI(
        model_name=llm_model,
        token=token,
        temperature=temp,
        max_new_tokens=256,
        request_timeout=timeout,
        provider = "featherless-ai",
    )
    return llm

def get_llm_model(llm_model: str = LLM, temp: float=TEMPERATURE, timeout: int=TIMEOUT):
    llm = Ollama(
        model=llm_model, 
        temperature=temp, 
        request_timeout=timeout)
    return llm

def _resolve_google_token() -> str:
    """Resolve Google API key from environment."""
    token = os.environ.get("GOOGLE_API_KEY")
    if not token:
        raise ValueError(
            "Missing Google API key. Set GOOGLE_API_KEY in .env."
        )
    return token

def get_gemini_llm(model_name: str = LLM, temp: float=TEMPERATURE, timeout: int=TIMEOUT):
    """Return a Gemini-backed LLM via LlamaIndex.

    Parameters
    ----------
    model_name: str
        Name of the Gemini model to use, e.g. "gemini-1.5-flash".
    temp: float
        Sampling temperature; defaults to CONFIG["TEMPERATURE"].
    timeout: int
        Request timeout in seconds; defaults to CONFIG["TIMEOUT"].
    """
    token = _resolve_google_token()
    llm = Gemini(
        model=model_name,
        api_key=token,
        temperature=temp,
        request_timeout=timeout,
    )
    return llm

def _resolve_groq_token() -> str:
    """Resolve Groq API key from environment."""
    token = os.environ.get("GROQ_API_KEY")
    if not token:
        raise ValueError(
            "Missing Groq API key. Set GROQ_API_KEY in .env."
        )
    return token

def get_groq_llm(model_name: str = LLM, temp: float=TEMPERATURE, timeout: int=TIMEOUT):
    """Return a Groq-backed LLM via LlamaIndex.

    Parameters
    ----------
    model_name: str
        Name of the Groq model to use, e.g. "llama3-8b-8192".
    temp: float
        Sampling temperature; defaults to CONFIG["TEMPERATURE"].
    timeout: int
        Request timeout in seconds; defaults to CONFIG["TIMEOUT"].
    """
    if Groq is None:
        raise ImportError(
            "Groq LLM backend is not available. Install 'llama-index-llms-groq'."
        )
    token = _resolve_groq_token()
    llm = Groq(
        model=model_name,
        api_key=token,
        temperature=temp,
        request_timeout=timeout,
    )
    return llm

def get_hf_inference_client(provider: str | None = None) -> InferenceClient:
    """Create a raw huggingface_hub InferenceClient using the resolved token.

    Parameters
    ----------
    provider: Optional provider string (e.g., "featherless-ai"). If not provided,
        defaults to the huggingface endpoint.
    """
    token = _resolve_hf_token()
    if provider:
        return InferenceClient(provider=provider, api_key=token)
    return InferenceClient(api_key=token)