import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = words[start : start + chunk_size]
        chunks.append(" ".join(chunk))
        start += (chunk_size - overlap)
    return chunks

def create_faiss_index(docs, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, embeddings, model

def retrieve_relevant_chunks(query, index, embeddings, docs, model_sbert, top_k=3):
    query_embedding = model_sbert.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    distances, indices = index.search(query_embedding, top_k)
    best_chunks = [docs[idx] for idx in indices[0]]
    return best_chunks
