import os

from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

from ..config import CONFIG


# configs
TEMPERATURE = CONFIG["TEMPERATURE"]
TIMEOUT = CONFIG["TIMEOUT"]


# load env variables
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")

AZURE_OPENAI_MODEL_DEPLOYMENT = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


def get_embedding_model(embed_model_id: str):
    Settings.embed_model = AzureOpenAIEmbedding(
        model=embed_model_id,
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_EMBEDDING_API_VERSION,
    )
    return Settings.embed_model

def get_llm_model(
    model_id: str,
    model_temperature: float = TEMPERATURE,
    model_request_timeout: int = TIMEOUT,
):
    Settings.llm = AzureOpenAI(
        model=model_id,
        engine=AZURE_OPENAI_MODEL_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        
        temperature=model_temperature,
        request_timeout=model_request_timeout,
    )
    return Settings.llm