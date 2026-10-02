from __future__ import annotations

import re
from functools import lru_cache
from html import unescape
from pathlib import Path
from typing import Dict, List

import nltk
import spacy
from bs4 import BeautifulSoup
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from .config import get_settings

NLTK_PACKAGES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def _bootstrap_nltk(data_dir: Path) -> None:
    if str(data_dir) not in nltk.data.path:
        nltk.data.path.append(str(data_dir))
    data_dir.mkdir(parents=True, exist_ok=True)
    for package, locator in NLTK_PACKAGES.items():
        try:
            nltk.data.find(locator)
        except LookupError:
            logger.info("Downloading NLTK resource: {}", package)
            nltk.download(package, download_dir=str(data_dir))


@lru_cache()
def _load_spacy_model() -> spacy.language.Language:
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model not found, falling back to blank 'en' pipeline.")
        return spacy.blank("en")


class Preprocessor:
    def __init__(self) -> None:
        settings = get_settings()
        _bootstrap_nltk(settings.data.nltk_data_dir)
        self.stopwords = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.spacy_nlp = _load_spacy_model()
        self.sentence_splitter = nltk.data.load("tokenizers/punkt/english.pickle")

    @staticmethod
    def clean_text(text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
        text = unescape(text)
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def sentence_segment(self, text: str) -> List[str]:
        cleaned = self.clean_text(text)
        return [sent.strip() for sent in self.sentence_splitter.tokenize(cleaned) if sent.strip()]

    def tokenize(self, sentence: str) -> List[str]:
        tokens = word_tokenize(sentence)
        return [tok.lower() for tok in tokens if tok.isalpha()]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [tok for tok in tokens if tok not in self.stopwords]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        return [self.lemmatizer.lemmatize(tok) for tok in tokens]

    def normalize_sentence(self, sentence: str) -> str:
        tokens = self.tokenize(sentence)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)

    def preprocess_document(self, document: str) -> Dict[str, List[str]]:
        sentences = self.sentence_segment(document)
        normalized = [self.normalize_sentence(sent) for sent in sentences]
        return {"sentences": sentences, "normalized_sentences": normalized}


