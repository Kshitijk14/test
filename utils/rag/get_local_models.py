from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ..config import CONFIG


TEMPERATURE = CONFIG["TEMPERATURE"]
TIMEOUT = CONFIG["TIMEOUT"]


def get_embedding_model(embed_model_id: str):
    embed_model = HuggingFaceEmbedding(model_name=embed_model_id)
    return embed_model

def get_llm_model(
    model_id: str, 
    temp: float=TEMPERATURE, 
    timeout: int=TIMEOUT
):
    llm = Ollama(
        model=model_id, 
        temperature=temp, 
        request_timeout=timeout)
    return llm
