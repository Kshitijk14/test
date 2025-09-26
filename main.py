from utils.rag.get_models import get_llm_model, get_gemini_llm , get_llm_model_hf , get_groq_llm
from rag import build_rag
from utils.config import CONFIG

ques = CONFIG["QUERY"]

llm = get_groq_llm()
response = build_rag(question=ques,llm=llm)
print(response)