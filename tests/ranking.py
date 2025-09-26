import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def rerank_chunks(query, chunks, embed_model):
    # Step 1: Generate embeddings for the query and the chunks
    query_embedding = embed_model.embed_query(query)
    
    # Step 2: Generate embeddings for each chunk
    chunk_embeddings = [embed_model.embed_query(chunk['text']) for chunk in chunks]
    
    # Step 3: Compute cosine similarities between the query and each chunk
    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    
    # Step 4: Pair each chunk with its similarity score and sort by similarity
    scored_chunks = [(similarity, chunk) for similarity, chunk in zip(similarities, chunks)]
    scored_chunks.sort(reverse=True, key=lambda x: x[0])  
    
    # Step 5: Return the chunks sorted by similarity
    return [chunk for _, chunk in scored_chunks]