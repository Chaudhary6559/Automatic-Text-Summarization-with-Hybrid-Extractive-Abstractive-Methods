from __future__ import annotations

from typing import List, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from .config import PostProcessingConfig


class SummaryPostProcessor:
    def __init__(self, config: PostProcessingConfig) -> None:
        self.config = config
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    @staticmethod
    def _remove_trigram_repetition(text: str) -> str:
        tokens = text.split()
        seen = set()
        filtered = []
        for i in range(len(tokens)):
            trigram = tuple(tokens[max(0, i - 2) : i + 1])
            if len(trigram) == 3:
                if trigram in seen:
                    continue
                seen.add(trigram)
            filtered.append(tokens[i])
        return " ".join(filtered)

    def _deduplicate_sentences(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return sentences
        embeddings = self.embedder.encode(sentences, convert_to_numpy=True)
        keep: List[str] = []
        keep_embeddings: List[np.ndarray] = []
        for idx, sentence in enumerate(sentences):
            if idx == 0:
                keep.append(sentence)
                keep_embeddings.append(embeddings[idx])
                continue
            if keep_embeddings:
                similarities = cosine_similarity(
                    [embeddings[idx]], np.array(keep_embeddings)
                )
                max_sim = float(similarities.max())
            else:
                max_sim = 0.0
            if max_sim < self.config.repetition_threshold:
                keep.append(sentence)
                keep_embeddings.append(embeddings[idx])
        return keep

    def enforce_length(self, document: str, summary: str) -> str:
        target_tokens = max(int(len(document.split()) * self.config.target_length_ratio), 1)
        summary_tokens = summary.split()
        if len(summary_tokens) <= target_tokens:
            return summary
        return " ".join(summary_tokens[:target_tokens])

    def enhance(self, document: str, summary: str) -> str:
        updated = summary
        if self.config.trigram_block:
            updated = self._remove_trigram_repetition(updated)
        sentences = [sent.strip() for sent in updated.split(". ") if sent.strip()]
        sentences = self._deduplicate_sentences(sentences)
        updated = ". ".join(sentences)
        updated = self.enforce_length(document, updated)
        logger.debug("Post-processed summary length: {}", len(updated.split()))
        return updated.strip()

