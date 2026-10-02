from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    document: str = Field(..., min_length=20, description="Raw document text to summarize.")
    top_k: Optional[int] = Field(None, description="Override number of extracted sentences.")
    max_length: Optional[int] = Field(None, description="Override max length for abstractive summary.")
    min_length: Optional[int] = Field(None, description="Override min length for abstractive summary.")
    return_extractive_sentences: bool = Field(
        default=True, description="Include the sentences selected by the extractive module."
    )
    run_postprocess: bool = Field(default=True, description="Apply repetition removal and coherence tweaks.")


class SummaryResponse(BaseModel):
    summary: str
    extractive_sentences: Optional[List[str]] = None
    compression_ratio: float
    timings: dict


class EvaluateRequest(BaseModel):
    generated_summary: str
    reference_summary: str
    metrics: Optional[List[str]] = None


class EvaluateResponse(BaseModel):
    scores: dict


class FileUploadResponse(BaseModel):
    extracted_text: str
    metadata: dict
    images_extracted: List[dict] = []
    filename: str


