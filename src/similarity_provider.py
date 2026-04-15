# loading text

print("Loading transformers...")

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

print("Loading torch...")
import torch
import numpy

from typing import TypedDict, cast

class SimilarityComputationResult(TypedDict):
    confident: bool
    similarities: list[float]

class SimilarityProvider:
    
    def __init__(self, model_name: str = "thenlper/gte-base", confident_threshold: float = 0.7, ambiguity_threshold: float = 0.07):
        
        self.model_name = model_name

        # confident measurement
        self.confident_threshold = confident_threshold
        self.ambiguity_threshold = ambiguity_threshold
        
        self._load_model()
        
    def _load_model(self):
        
        print(f"Loading transformer model for '{self.model_name}'")
        self.model = SentenceTransformer(self.model_name)
        print(f"Loading tokenizer for '{self.model_name}'")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def compute_embeddings(self, word: str) -> "numpy.ndarray":
        target_vector: "numpy.ndarray" = self.model.encode(word.strip(), convert_to_numpy=True)
        
        return target_vector
    
    def compute_similarity(self, word: str, phrases: list[str]) -> SimilarityComputationResult:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        target_vector = self.model.encode(word.strip(), convert_to_tensor=True)
        phrase_vectors = self.model.encode([p.strip() for p in phrases], convert_to_tensor=True)
        
        similarities = torch.nn.functional.cosine_similarity(target_vector, phrase_vectors).tolist()
        sorted_similarities = sorted(similarities, reverse=True)
        
        is_confident: bool = max(similarities) > self.confident_threshold
        if len(sorted_similarities) > 1 and (sorted_similarities[0] - sorted_similarities[1]) < self.ambiguity_threshold:
            is_confident: bool = False
        
        result: SimilarityComputationResult = cast(SimilarityComputationResult, {
            "confident": cast(bool, is_confident),
            "similarities": cast(list[float], similarities)
        }) # shut up
        
        print(f"result: {result}")
        
        return result
        
        
        