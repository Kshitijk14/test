import os
import chromadb
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter

from utils.config import CONFIG
from utils.rag.get_local_models import get_embedding_model
from utils.rag.get_remote_models import get_llm_model
from utils.rag.populate import get_chunk_id
from utils.rag.rerank import rerank_chunks
from utils.rag.get_prompt import RESPONSE_SYNTHESIS_PROMPT


# Initialize FastAPI app
app = FastAPI(
    title="RAG Backend",
    description="FastAPI backend for RAG with lazy loading and re-ranking",
    version="1.0.0"
)

# Global variables for lazy loading
_rag_state = {
    "initialized": False,
    "query_engine": None,
    "retriever": None,
    "embed_model": None,
    "llm": None,
}

# Config
DATA_DIR = Path(CONFIG["DATA_DIR"])
CHROMA_DIR = Path(CONFIG["CHROMA_DIR"])
CHROMA_COLLECTION = CONFIG["CHROMA_COLLECTION"]

LOCAL_EMBEDDING_MODEL = CONFIG["LOCAL_EMBEDDING_MODEL"]
LOCAL_LLM_MODEL = CONFIG["LOCAL_LLM_MODEL"]

REMOTE_EMBEDDING_MODEL = CONFIG["REMOTE_EMBEDDING_MODEL"]
REMOTE_LLM_MODEL = CONFIG["REMOTE_LLM_MODEL"]

CHUNK_SIZE = CONFIG["CHUNK_SIZE"]
CHUNK_OVERLAP = CONFIG["CHUNK_OVERLAP"]
TOP_K = CONFIG["TOP_K"]
TOP_N = CONFIG["TOP_N"]


# Request/Response models
class QueryRequest(BaseModel):
    query: str


class RankedChunk(BaseModel):
    chunk_id: str
    file_name: str
    score: float


class QueryResponse(BaseModel):
    query: str
    response: str
    retrieved_chunks: list[RankedChunk]
    top_n_chunks: list[RankedChunk]


def initialize_rag():
    """Lazy initialization of RAG components."""
    if _rag_state["initialized"]:
        return

    print("Initializing RAG components...")

    # 1. Load PDF/DOC/TXT documents
    os.makedirs(DATA_DIR, exist_ok=True)
    docs = SimpleDirectoryReader(DATA_DIR).load_data()

    # 2. Split docs into chunks
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    nodes = []
    for doc in docs:
        file_name = doc.metadata.get("file_name", "doc")
        doc_nodes = splitter.get_nodes_from_documents([doc])
        # assign stable IDs
        for idx, node in enumerate(doc_nodes):
            node.node_id = get_chunk_id(node.text, file_name, idx)
        nodes.extend(doc_nodes)

    # 3. Setup embedding + LLM
    embed_model = get_embedding_model(LOCAL_EMBEDDING_MODEL)
    llm = get_llm_model(REMOTE_LLM_MODEL)

    # 4. Setup persistent Chroma client
    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 5. Wrap vector_store with StorageContext
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 6. Apply global settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    # 7. Build or update index
    existing_ids = set(chroma_collection.get()["ids"])
    new_nodes = [n for n in nodes if n.node_id not in existing_ids]

    if new_nodes:
        print(f"Adding {len(new_nodes)} new chunks to Chroma...")
        # Add only new chunks
        index = VectorStoreIndex(
            new_nodes,
            storage_context=storage_context,
        )
    else:
        print("No new chunks to add. Using existing index.")
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    # 8. Define a custom prompt
    qa_template = RESPONSE_SYNTHESIS_PROMPT

    # 9. Create query engine with prompt
    query_engine = index.as_query_engine(
        text_qa_template=qa_template,
        similarity_top_k=TOP_K,
    )

    # 10. Create retriever
    retriever = index.as_retriever(similarity_top_k=TOP_K)

    # Store in global state
    _rag_state["query_engine"] = query_engine
    _rag_state["retriever"] = retriever
    _rag_state["embed_model"] = embed_model
    _rag_state["llm"] = llm
    _rag_state["initialized"] = True

    print("RAG initialization complete!")


@app.on_event("startup")
async def startup_event():
    """Initialize RAG on startup."""
    initialize_rag()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_initialized": _rag_state["initialized"]
    }


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system with re-ranking."""
    if not _rag_state["initialized"]:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        query = request.query

        # 1. Get query engine and retriever
        query_engine = _rag_state["query_engine"]
        retriever = _rag_state["retriever"]

        # 2. Retrieve top_k chunks
        retrieved_nodes = retriever.retrieve(query)

        # 3. Extract chunk texts and metadata
        chunk_texts = [node.get_content() for node in retrieved_nodes]
        chunk_metadata = [
            {
                "chunk_id": node.node_id,
                "file_name": node.metadata.get("file_name", "unknown_file"),
                "content": node.get_content(),
            }
            for node in retrieved_nodes
        ]

        # 4. Re-rank chunks using cross-encoder
        reranked_chunks_with_scores = rerank_chunks(chunk_texts, [query])

        # 5. Keep only top_n re-ranked results
        top_n_chunks_with_scores = reranked_chunks_with_scores[:TOP_N]

        # 6. Map scores back to metadata
        retrieved_chunks_response = [
            RankedChunk(
                chunk_id=meta["chunk_id"],
                file_name=meta["file_name"],
                score=next(
                    (score for chunk, score in reranked_chunks_with_scores 
                     if chunk == meta["content"]),
                    0.0
                )
            )
            for meta in chunk_metadata
        ]

        top_n_chunks_response = [
            RankedChunk(
                chunk_id=next(
                    (meta["chunk_id"] for meta in chunk_metadata 
                     if meta["content"] == chunk),
                    "unknown"
                ),
                file_name=next(
                    (meta["file_name"] for meta in chunk_metadata 
                     if meta["content"] == chunk),
                    "unknown"
                ),
                score=score
            )
            for chunk, score in top_n_chunks_with_scores
        ]

        # 7. Generate final response using query engine
        response = query_engine.query(query)

        return QueryResponse(
            query=query,
            response=str(response),
            retrieved_chunks=retrieved_chunks_response,
            top_n_chunks=top_n_chunks_response
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get RAG system statistics."""
    if not _rag_state["initialized"]:
        return {"status": "not_initialized"}

    return {
        "status": "initialized",
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "top_n": TOP_N,
            "embedding_model": LOCAL_EMBEDDING_MODEL,
            "llm_model": REMOTE_LLM_MODEL,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
