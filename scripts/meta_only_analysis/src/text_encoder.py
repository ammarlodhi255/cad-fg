import os
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path

class TextFeatureEncoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", cache_dir="outputs/text_embeds"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = SentenceTransformer(model_name)
        self.embed_dim = self.model.get_sentence_embedding_dimension()
    
    def _get_cache_path(self, text, prefix=""):
        """Get cache file path for a text string"""
        # Simple hash for filename
        text_hash = str(abs(hash(text)))[:10]
        return self.cache_dir / f"{prefix}_{text_hash}.pt"
    
    def encode(self, texts, prefix=""):
        """
        Encode a list of texts, using cache when available
        
        Args:
            texts: List of strings to encode
            prefix: Cache filename prefix
        Returns:
            torch.Tensor of shape (len(texts), embed_dim)
        """
        if not isinstance(texts, list):
            texts = [texts]
        
        # Initialize output tensor
        embeddings = []
        texts_to_encode = []
        cache_paths = []
        
        # Check cache first
        for text in texts:
            if not text or pd.isna(text):
                # Handle empty/nan text
                embeddings.append(torch.zeros(self.embed_dim))
                continue
                
            cache_path = self._get_cache_path(text, prefix)
            if cache_path.exists():
                embedding = torch.load(cache_path)
                embeddings.append(embedding)
            else:
                texts_to_encode.append(text)
                cache_paths.append(cache_path)
        
        # Encode new texts
        if texts_to_encode:
            new_embeddings = self.model.encode(
                texts_to_encode,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Cache and collect
            for emb, cache_path in zip(new_embeddings, cache_paths):
                torch.save(emb, cache_path)
                embeddings.append(emb)
        
        return torch.stack(embeddings)

def process_text_features(texts, text_encoder):
    """
    Process multiple text fields into a single embedding
    
    Args:
        texts: Dict of text field names to values
        text_encoder: TextFeatureEncoder instance
    Returns:
        torch.Tensor of concatenated embeddings
    """
    embeddings = []
    for field, text in texts.items():
        # Encode with field name as cache prefix
        emb = text_encoder.encode(text, prefix=field)
        embeddings.append(emb)
    
    # Concatenate all text embeddings
    return torch.cat(embeddings, dim=-1)
