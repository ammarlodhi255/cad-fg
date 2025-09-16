# src/text/encoders.py
import os, torch
from typing import List, Optional
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError("Please `pip install sentence-transformers`") from e


def load_sentence_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                          device: Optional[str] = None):
    """
    Returns a SentenceTransformer with .encode(texts, convert_to_tensor=True).
    """
    m = SentenceTransformer(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(device)
    return m


def encode_texts(texts: List[str], model, batch_size: int = 64):
    """
    texts: list of str. Returns a torch.Tensor [N, D] on CPU.
    """
    with torch.inference_mode():
        emb = model.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
    return emb.detach().cpu()