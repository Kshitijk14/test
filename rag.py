import os
from pathlib import Path
import chromadb

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
    PromptTemplate
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter

from utils.config import CONFIG
from utils.rag.get_models import get_embedding_model, get_llm_model
from utils.rag.populate import get_chunk_id
from utils.rag.get_prompt import RESPONSE_SYNTHESIS_PROMPT
from tests.ranking import rerank_chunks

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

QUERY = CONFIG["QUERY"]

# # tracing
# endpoint = "http://localhost:6006/v1/traces"
# tracer_provider = trace_sdk.TracerProvider()
# tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
# LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

def build_rag():
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

    # 9. Create query engine with prompt
    query_engine = index.as_query_engine(
        text_qa_template=qa_template,
        similarity_top_k=TOP_K,
    )

    # 10. Run a query
    response = query_engine.query(QUERY)

    retrieved_chunks = [
        {"text": node.node.get_content(), "metadata": node.node.metadata}
        for node in response.source_nodes
    ]

    reranked_chunks = rerank_chunks(QUERY, retrieved_chunks, llm)

    top_chunks = reranked_chunks[:TOP_K]

    print("\n=== RE-RANKED CHUNKS ===\n")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"Ranked Chunk {i}: {chunk['text']}")
        print("-" * 60)

    # Final response based on the re-ranked chunks
    final_response = query_engine.query_with_chunks(QUERY, top_chunks)
    print("\n=== FINAL RESPONSE ===\n")
    print(final_response)


if __name__ == "__main__":
    build_rag()
