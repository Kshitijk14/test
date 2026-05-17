from sentence_transformers import CrossEncoder

# Initialize the cross-encoder model
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_chunks(chunks, queries):
    """Rerank the given chunks based on the queries using the cross-encoder model.
    
    Returns:
        list: List of tuples containing (chunk, score) sorted by score in descending order.
    """
    scores = model.predict([(query, chunk) for query in queries for chunk in chunks])
    ranked_indices = scores.argsort()[::-1]  # Descending order
    return [(chunks[idx], scores[idx]) for idx in ranked_indices]