from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .src.config import get_settings
from .src.file_extractor import FileExtractor
from .src.hybrid_summarizer import HybridSummarizer
from .src.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    FileUploadResponse,
    SummarizeRequest,
    SummaryResponse,
)

settings = get_settings()
app = FastAPI(
    title="Hybrid Extractive-Abstractive Summarization API",
    description="REST API for the hybrid summarization project.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize summarizer lazily to handle import errors gracefully
summarizer: HybridSummarizer | None = None
file_extractor: FileExtractor | None = None


def get_summarizer() -> HybridSummarizer:
    """Get or initialize the summarizer."""
    global summarizer
    if summarizer is None:
        try:
            summarizer = HybridSummarizer()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Summarizer initialization failed: {str(e)}. Please check model downloads and dependencies.",
            )
    return summarizer


def get_file_extractor() -> FileExtractor:
    """Get or initialize the file extractor."""
    global file_extractor
    if file_extractor is None:
        try:
            # Use EasyOCR for better accuracy (can be slower)
            # Set to False to use faster pytesseract
            file_extractor = FileExtractor(use_easyocr=False)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"File extractor initialization failed: {str(e)}. Please check OCR dependencies.",
            )
    return file_extractor


@app.get("/")
def root() -> dict:
    """Root endpoint with API information."""
    return {
        "message": "Hybrid Extractive-Abstractive Summarization API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "summarize": "/summarize",
            "summarize_file": "/summarize-file",
            "extract_text": "/extract-text",
            "evaluate": "/evaluate",
            "docs": "/docs",
            "redoc": "/redoc",
        },
        "supported_formats": [
            "PDF (.pdf)",
            "Word Documents (.docx, .doc)",
            "Text Files (.txt)",
            "Markdown (.md)",
            "Images (.png, .jpg, .jpeg, .gif, .bmp, .tiff, .webp) - with OCR",
        ],
    }


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/summarize", response_model=SummaryResponse)
def summarize(request: SummarizeRequest) -> SummaryResponse:
    s = get_summarizer()
    result = s.summarize(
        request.document,
        top_k=request.top_k,
        max_length=request.max_length,
        min_length=request.min_length,
        run_postprocess=request.run_postprocess,
    )
    return SummaryResponse(
        summary=result.summary,
        extractive_sentences=result.extractive_sentences if request.return_extractive_sentences else None,
        compression_ratio=result.compression_ratio,
        timings=result.timings,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    s = get_summarizer()
    scores = s.evaluate(
        generated=request.generated_summary,
        reference=request.reference_summary,
        metrics=request.metrics,
    )
    return EvaluateResponse(scores=scores)


@app.post("/extract-text", response_model=FileUploadResponse)
async def extract_text(file: UploadFile = File(...)) -> FileUploadResponse:
    """
    Extract text from uploaded file (PDF, DOCX, TXT, MD, images, etc.).
    Supports OCR for images and embedded images in documents.
    """
    try:
        extractor = get_file_extractor()
        file_content = await file.read()

        result = extractor.extract_text(
            file_content=file_content,
            filename=file.filename or "unknown",
            mime_type=file.content_type,
        )

        return FileUploadResponse(
            extracted_text=result["text"],
            metadata=result["metadata"],
            images_extracted=result["images_extracted"],
            filename=file.filename or "unknown",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")


@app.post("/summarize-file", response_model=SummaryResponse)
async def summarize_file(
    file: UploadFile = File(...),
    top_k: int | None = None,
    max_length: int | None = None,
    min_length: int | None = None,
    run_postprocess: bool = True,
    return_extractive_sentences: bool = True,
) -> SummaryResponse:
    """
    Extract text from file and summarize it in one step.
    Supports PDF, DOCX, TXT, MD, images, etc. with OCR.
    """
    try:
        # Extract text from file
        extractor = get_file_extractor()
        file_content = await file.read()

        extraction_result = extractor.extract_text(
            file_content=file_content,
            filename=file.filename or "unknown",
            mime_type=file.content_type,
        )

        extracted_text = extraction_result["text"]

        if len(extracted_text.strip()) < 20:
            raise HTTPException(
                status_code=400,
                detail="Extracted text is too short. Please ensure the file contains readable text or images with text.",
            )

        # Summarize the extracted text
        s = get_summarizer()
        result = s.summarize(
            extracted_text,
            top_k=top_k,
            max_length=max_length,
            min_length=min_length,
            run_postprocess=run_postprocess,
        )

        return SummaryResponse(
            summary=result.summary,
            extractive_sentences=result.extractive_sentences if return_extractive_sentences else None,
            compression_ratio=result.compression_ratio,
            timings=result.timings,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


