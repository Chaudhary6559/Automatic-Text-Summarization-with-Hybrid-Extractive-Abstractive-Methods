from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence

import networkx as nx
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer, util

from .config import ExtractiveConfig
from .feature_extraction import build_sentence_features
from .preprocessing import Preprocessor


def _textrank_scores(similarity_matrix: np.ndarray, damping: float) -> np.ndarray:
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph, alpha=damping)
    ranked = np.array([scores.get(idx, 0.0) for idx in range(len(similarity_matrix))])
    return ranked / (ranked.sum() + 1e-9)


def _mmr_selection(
    scores: np.ndarray,
    embeddings: np.ndarray,
    sentences: Sequence[str],
    top_k: int,
    lambda_param: float,
) -> List[int]:
    if len(sentences) <= top_k:
        return list(range(len(sentences)))

    selected: List[int] = []
    candidate_indices = list(range(len(sentences)))
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)

    while len(selected) < top_k and candidate_indices:
        mmr_scores = []
        for idx in candidate_indices:
            sim_to_selected = 0.0
            if selected:
                sim_to_selected = max(
                    float(np.dot(embeddings[idx], embeddings[s_sel])) for s_sel in selected
                )
            mmr_score = lambda_param * scores[idx] - (1 - lambda_param) * sim_to_selected
            mmr_scores.append((mmr_score, idx))
        _, best_idx = max(mmr_scores, key=lambda item: item[0])
        selected.append(best_idx)
        candidate_indices.remove(best_idx)
    return sorted(selected)


def _token_distribution(tokens: List[str]) -> dict[str, float]:
    counter = Counter(tokens)
    total = sum(counter.values()) or 1
    return {token: count / total for token, count in counter.items()}


def _kl_divergence(doc_dist: dict[str, float], sent_dist: dict[str, float]) -> float:
    eps = 1e-9
    divergence = 0.0
    for token, p_doc in doc_dist.items():
        p_sent = sent_dist.get(token, eps)
        divergence += p_doc * np.log((p_doc + eps) / (p_sent + eps))
    return divergence


@dataclass
class ExtractiveResult:
    sentences: List[str]
    indices: List[int]
    scores: List[float]


class BERTExtractiveSummarizer:
    def __init__(self, config: ExtractiveConfig, preprocessor: Preprocessor) -> None:
        self.config = config
        self.preprocessor = preprocessor
        logger.info("Loading SentenceTransformer model: {}", config.embedding_model)
        self.embedder = SentenceTransformer(config.embedding_model)

    def summarize(self, document: str, top_k: int | None = None) -> ExtractiveResult:
        processed = self.preprocessor.preprocess_document(document)
        sentences = processed["sentences"][: self.config.max_sentences]
        if not sentences:
            raise ValueError("Document does not contain valid sentences.")

        normalized_sentences = processed["normalized_sentences"][: len(sentences)]
        document_tokens = []
        for norm in normalized_sentences:
            document_tokens.extend(norm.split())
        doc_distribution = _token_distribution(document_tokens)

        embeddings = self.embedder.encode(sentences, convert_to_numpy=True)
        similarity = util.cos_sim(embeddings, embeddings).cpu().numpy()
        np.fill_diagonal(similarity, 0.0)
        similarity = np.maximum(similarity, 0.0)

        textrank = _textrank_scores(similarity, self.config.textrank_damping)
        features = build_sentence_features(sentences, self.preprocessor)

        kl_penalties = []
        for norm_sentence in normalized_sentences:
            sentence_dist = _token_distribution(norm_sentence.split())
            kl_penalties.append(_kl_divergence(doc_distribution, sentence_dist))
        kl_penalties = np.array(kl_penalties)
        kl_scores = 1 - (kl_penalties - kl_penalties.min()) / (kl_penalties.max() - kl_penalties.min() + 1e-9)

        combined_score = (
            0.5 * textrank
            + self.config.position_bias * features["position"]
            + 0.3 * features["centroid"]
            + 0.2 * features["ner"]
            + 0.15 * features["length"]
            + 0.05 * features["cue"]
            + 0.2 * kl_scores
        )

        token_count = sum(len(sent.split()) for sent in sentences)
        dynamic_top_k = (
            self.config.top_k_short if token_count < 400 else self.config.top_k_long
        )
        if top_k:
            dynamic_top_k = top_k

        indices = _mmr_selection(
            combined_score,
            embeddings,
            sentences,
            top_k=dynamic_top_k,
            lambda_param=self.config.redundancy_lambda,
        )
        selected_sentences = [sentences[idx] for idx in indices]
        selected_scores = [float(combined_score[idx]) for idx in indices]
        return ExtractiveResult(
            sentences=selected_sentences,
            indices=indices,
            scores=selected_scores,
        )

