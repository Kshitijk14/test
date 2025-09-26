print("rag.py")

import os
from pathlib import Path
import chromadb

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
)

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank

from utils.config import CONFIG
from utils.rag.get_models import get_embedding_model, get_llm_model, get_llm_model_hf
from utils.rag.populate import get_chunk_id
from utils.rag.get_prompt import RESPONSE_SYNTHESIS_PROMPT

# import phoenix as px
# from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# from opentelemetry.sdk import trace as trace_sdk
# from opentelemetry.sdk.trace.export import SimpleSpanProcessor


# configs
DATA_DIR = Path(CONFIG["DATA_DIR"])
CHROMA_DIR = Path(CONFIG["CHROMA_DIR"])
CHROMA_COLLECTION = CONFIG["CHROMA_COLLECTION"]

EMBEDDING_MODEL = CONFIG["EMBEDDING_MODEL"]
LLM_MODEL = CONFIG["LLM_MODEL"]

CHUNK_SIZE = CONFIG["CHUNK_SIZE"]
CHUNK_OVERLAP = CONFIG["CHUNK_OVERLAP"]
TOP_K = CONFIG["TOP_K"]

# Re-ranker configuration
RERANKER_MODEL = CONFIG["RERANKER_MODEL"]
RERANKER_TOP_N = CONFIG["RERANKER_TOP_N"]

QUERY = CONFIG["QUERY"]

# # tracing
# tracer_provider = trace_sdk.TracerProvider()
# tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
# LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


def build_rag(question: str, llm=None):
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
    embed_model = get_embedding_model(embedding_model=EMBEDDING_MODEL)
    # Allow injection (e.g., Ollama) while keeping default HF path
    if llm is None:
        llm = get_llm_model(llm_model=LLM_MODEL)

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

    # 9. Create re-ranker for better chunk relevance
    # Using a cross-encoder model to re-rank retrieved chunks based on query similarity
    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL,
        top_n=RERANKER_TOP_N,  # Keep top N most relevant chunks after re-ranking
    )

    # 10. Create query engine with re-ranker
    query_engine = index.as_query_engine(
        text_qa_template=qa_template,
        similarity_top_k=TOP_K,
        node_postprocessors=[reranker],  # Add re-ranker to the pipeline
    )

    # 11. Run a query
    q = question if question is not None else QUERY
    response = query_engine.query(q)
    
    
    print("\n=== RETRIEVED CHUNKS ===\n")
    for i, node in enumerate(response.source_nodes, 1):
        file_name = node.node.metadata.get("file_name", "unknown_file")
        chunk_id = node.node.node_id
        
        print(f"Chunk {i} | ID: {chunk_id} | File: {file_name}")
        
        # print(node.node.get_content())
        print("-" * 60)
    
    print("\n=== RESPONSE ===\n")
    return response

if __name__ == "__main__":
    build_rag(question=None)