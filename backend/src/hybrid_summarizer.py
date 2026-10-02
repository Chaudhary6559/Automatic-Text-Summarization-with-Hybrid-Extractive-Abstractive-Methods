from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

from .abstractive_summarizer import BARTAbstractiveSummarizer
from .config import AbstractiveConfig, ExtractiveConfig, PostProcessingConfig, get_settings
from .evaluation import evaluate_summary
from .extractive_summarizer import BERTExtractiveSummarizer
from .postprocessing import SummaryPostProcessor
from .preprocessing import Preprocessor


@dataclass
class HybridSummary:
    summary: str
    extractive_sentences: list[str]
    timings: dict[str, float]
    compression_ratio: float


class HybridSummarizer:
    def __init__(
        self,
        extractive_cfg: ExtractiveConfig | None = None,
        abstractive_cfg: AbstractiveConfig | None = None,
        post_cfg: PostProcessingConfig | None = None,
    ) -> None:
        settings = get_settings()
        self.preprocessor = Preprocessor()
        self.extractive = BERTExtractiveSummarizer(extractive_cfg or settings.extractive, self.preprocessor)
        self.abstractive = BARTAbstractiveSummarizer(abstractive_cfg or settings.abstractive)
        self.postprocessor = SummaryPostProcessor(post_cfg or settings.postprocessing)

    def summarize(
        self,
        document: str,
        *,
        top_k: int | None = None,
        max_length: int | None = None,
        min_length: int | None = None,
        run_postprocess: bool = True,
    ) -> HybridSummary:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        extractive_result = self.extractive.summarize(document, top_k=top_k)
        timings["extractive"] = round(time.perf_counter() - start, 4)

        start = time.perf_counter()
        extractive_text = " ".join(extractive_result.sentences)
        abstractive_result = self.abstractive.generate(
            extractive_text,
            max_length=max_length,
            min_length=min_length,
        )
        summary_text = abstractive_result.summary
        timings["abstractive"] = round(time.perf_counter() - start, 4)

        if run_postprocess:
            start = time.perf_counter()
            summary_text = self.postprocessor.enhance(document, summary_text)
            timings["postprocessing"] = round(time.perf_counter() - start, 4)

        compression_ratio = len(summary_text.split()) / max(len(document.split()), 1)
        logger.info("Generated summary with compression ratio {:.3f}", compression_ratio)

        return HybridSummary(
            summary=summary_text,
            extractive_sentences=extractive_result.sentences,
            timings=timings,
            compression_ratio=round(compression_ratio, 4),
        )

    def evaluate(self, generated: str, reference: str, metrics: list[str] | None = None) -> dict:
        return evaluate_summary(generated, reference, metrics)


