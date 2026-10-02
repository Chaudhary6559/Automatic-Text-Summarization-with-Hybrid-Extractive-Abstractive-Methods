import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.src.feature_extraction import build_sentence_features
from backend.src.preprocessing import Preprocessor


def test_feature_builder_returns_expected_keys():
    preprocessor = Preprocessor()
    sentences = [
        "Hybrid summarization blends extractive and abstractive techniques.",
        "BERT selects salient sentences before BART rewrites them.",
    ]
    features = build_sentence_features(sentences, preprocessor)
    assert set(features.keys()) == {"length", "position", "cue", "ner", "centroid"}
    for value in features.values():
        assert len(value) == len(sentences)

