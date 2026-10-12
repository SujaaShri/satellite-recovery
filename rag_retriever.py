from sentence_transformers import SentenceTransformer
import numpy as np
from fault_knowledge_base import FAULT_KNOWLEDGE_BASE

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def _doc_text(doc):
    return f"{doc['title']}. {doc['content']}"

class FaultRetriever:
    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base or FAULT_KNOWLEDGE_BASE
        self.model = get_model()
        self.doc_embeddings = self.model.encode([_doc_text(d) for d in self.kb], convert_to_numpy=True)

    def retrieve_documents(self, query, top_k=3, fault_type_filter=None):
        candidates, embeddings = self.kb, self.doc_embeddings
        if fault_type_filter:
            idx = [i for i, d in enumerate(self.kb) if d["fault_type"] == fault_type_filter]
            if idx:
                candidates = [self.kb[i] for i in idx]
                embeddings = self.doc_embeddings[idx]
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        sims = embeddings @ query_emb / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [{**candidates[i], "similarity_score": float(sims[i])} for i in top_indices]