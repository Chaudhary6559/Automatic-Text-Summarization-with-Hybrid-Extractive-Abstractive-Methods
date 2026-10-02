from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import AbstractiveConfig


@dataclass
class AbstractiveResult:
    summary: str
    num_tokens: int


class BARTAbstractiveSummarizer:
    def __init__(self, config: AbstractiveConfig) -> None:
        self.config = config
        logger.info("Loading abstractive model: {}", config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        self.model.to(self.device)

    def generate(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> AbstractiveResult:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length,
        ).to(self.device)
        generation_kwargs = dict(
            num_beams=self.config.num_beams,
            length_penalty=self.config.length_penalty,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            early_stopping=self.config.early_stopping,
            max_length=max_length or self.config.max_output_length,
            min_length=min_length or self.config.min_output_length,
        )
        summary_ids = self.model.generate(**inputs, **generation_kwargs)
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return AbstractiveResult(summary=summary_text.strip(), num_tokens=len(summary_text.split()))

