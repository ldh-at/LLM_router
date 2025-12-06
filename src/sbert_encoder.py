import torch
from sentence_transformers import SentenceTransformer

class SBERTEncoder:
    def __init__(self, device="cuda"):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.device = device

    def encode(self, texts):
        return self.model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device
        )
