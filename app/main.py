from fastapi import FastAPI
from pydantic import BaseModel
from rag import build_rag
from utils.rag.get_models import get_groq_llm

llm = get_groq_llm()
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: str

@app.post("/query", response_model=QueryResponse)
def query_rag(data: QueryRequest):
    rag_response = build_rag(question=data.question, llm=llm)
    # Extract the string and wrap in Pydantic model
    return QueryResponse(response=rag_response.response)
