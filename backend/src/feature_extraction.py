from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import Preprocessor

CUE_PHRASES = {
    "in conclusion",
    "in summary",
    "overall",
    "however",
    "therefore",
    "consequently",
    "importantly",
}


def normalize(values: np.ndarray) -> np.ndarray:
    if values.std() == 0:
        return np.zeros_like(values)
    return (values - values.min()) / (values.max() - values.min() + 1e-9)


def build_sentence_features(sentences: List[str], preprocessor: Preprocessor) -> Dict[str, np.ndarray]:
    sentence_lengths = np.array([len(sent.split()) for sent in sentences], dtype=float)
    length_feature = normalize(sentence_lengths)

    positions = np.arange(len(sentences), dtype=float)
    position_feature = 1 - normalize(positions)  # earlier sentences receive higher scores

    cue_feature = np.array(
        [1.0 if any(phrase in sent.lower() for phrase in CUE_PHRASES) else 0.0 for sent in sentences],
        dtype=float,
    )

    ner_density = []
    for sent in sentences:
        doc = preprocessor.spacy_nlp(sent)
        if len(sent.split()) == 0:
            ner_density.append(0.0)
            continue
        ner_density.append(len(doc.ents) / len(sent.split()))
    ner_feature = normalize(np.array(ner_density, dtype=float))

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    centroid = tfidf_matrix.mean(axis=0)
    centroid_scores = normalize(
        np.asarray(tfidf_matrix @ centroid.T).reshape(-1)
    )

    return {
        "length": length_feature,
        "position": position_feature,
        "cue": cue_feature,
        "ner": ner_feature,
        "centroid": centroid_scores,
    }


