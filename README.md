# test

## Features

- **HuggingFace Inference API Integration**: Uses `meta-llama/Llama-3.1-8B-Instruct` via HF API
- **SentenceTransformer Re-ranker**: Improves chunk relevance using cross-encoder models
- **Configurable Parameters**: All settings in `params.yaml`
- **Persistent Vector Store**: ChromaDB for document storage and retrieval
- **Comprehensive Testing**: Pytest integration for LLM validation

## Re-ranker Details

The system now includes a **SentenceTransformerRerank** component that addresses the "empty response" issue by:

- **Cross-encoder scoring**: Uses `cross-encoder/ms-marco-TinyBERT-L-2-v2` to score chunk relevance
- **Top-N selection**: Keeps only the top 3 most relevant chunks (configurable via `RERANKER_TOP_N`)
- **Improved context**: Provides more focused and relevant context to the LLM
- **Better responses**: Reduces empty or irrelevant responses from the LLM

## rag:

1. setup:
```bash
uv sync
```

2. configure hugging face (no Ollama required):
Ensure your Hugging Face API token is available to the app. You can set it in `params.yaml` under `HUGGINGFACE_API_KEY` (already wired), or via environment variable:
```bash
$env:HUGGINGFACEHUB_API_TOKEN="<your_hf_token>"   # PowerShell
# or
set HUGGINGFACEHUB_API_TOKEN=<your_hf_token>       # cmd
```

3. tracing:
```bash
uv run -m phoenix.server.main serve
```

4. run:
```bash
uv run rag.py
```

5. tests:
```bash
uv run -m pytest -q
```


## task:

1. LLM returning an empty response, even when it was able to retrieve chunks & everything, why?
2. Use HF Inference API, instead of Ollama local model.
3. pytest?
4. Add re-ranker
5. FastAPI backend

---