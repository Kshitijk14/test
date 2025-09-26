from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

from ..config import CONFIG
import os 
from dotenv import load_dotenv
load_dotenv("/home/mayank/Documents/dollarstore/test/.env")

openai_api_key = os.getenv("OPENAI_API_KEY")

TEMPERATURE = CONFIG["TEMPERATURE"]
TIMEOUT = CONFIG["TIMEOUT"]


def get_embedding_model(embedding_model: str):
    embed_model = HuggingFaceEmbedding(model_name=embedding_model)
    return embed_model

def get_llm_model(llm_model: str, temp: float=TEMPERATURE, timeout: int=TIMEOUT):
    llm = OpenAI(
        model_name=llm_model, 
        temperature=temp, 
        openai_api_key=openai_api_key,  # Pass the OpenAI API Key
        request_timeout=timeout
    )
    return llm
