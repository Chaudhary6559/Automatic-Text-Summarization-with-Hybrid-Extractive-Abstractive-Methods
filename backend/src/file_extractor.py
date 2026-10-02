"""File extraction module for processing various document formats and images with OCR."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import markdown
import pdfplumber
import pytesseract
from docx import Document
from PIL import Image
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


class FileExtractor:
    """Extract text from various file formats including images with OCR."""

    SUPPORTED_EXTENSIONS = {
        ".txt": "text",
        ".md": "markdown",
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "docx",  # Will try to handle as docx
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".gif": "image",
        ".bmp": "image",
        ".tiff": "image",
        ".tif": "image",
        ".webp": "image",
    }

    def __init__(self, use_easyocr: bool = False):
        """
        Initialize file extractor.

        Args:
            use_easyocr: If True, use EasyOCR for better OCR accuracy (slower but better).
                        If False, use pytesseract (faster but less accurate).
        """
        self.use_easyocr = use_easyocr
        self._easyocr_reader = None

    def _get_easyocr_reader(self):
        """Lazy load EasyOCR reader."""
        if self._easyocr_reader is None:
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}. Falling back to pytesseract.")
                self.use_easyocr = False
        return self._easyocr_reader

    def extract_text(self, file_content: bytes, filename: str, mime_type: str | None = None) -> dict[str, Any]:
        """
        Extract text from file content.

        Args:
            file_content: Raw file bytes
            filename: Original filename (used to determine format)
            mime_type: Optional MIME type hint

        Returns:
            Dictionary with:
                - text: Extracted text
                - metadata: File metadata (pages, images found, etc.)
                - images_extracted: List of text extracted from images
        """
        file_path = Path(filename)
        extension = file_path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        file_type = self.SUPPORTED_EXTENSIONS[extension]

        try:
            if file_type == "text":
                return self._extract_text_file(file_content)
            elif file_type == "markdown":
                return self._extract_markdown(file_content)
            elif file_type == "pdf":
                return self._extract_pdf(file_content)
            elif file_type == "docx":
                return self._extract_docx(file_content)
            elif file_type == "image":
                return self._extract_image(file_content)
            else:
                raise ValueError(f"Unknown file type: {file_type}")
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {e}")
            raise ValueError(f"Failed to extract text from {filename}: {str(e)}")

    def _extract_text_file(self, content: bytes) -> dict[str, Any]:
        """Extract text from plain text file."""
        try:
            # Try UTF-8 first
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1
                text = content.decode("latin-1")
            except UnicodeDecodeError:
                # Last resort: ignore errors
                text = content.decode("utf-8", errors="ignore")

        return {
            "text": text,
            "metadata": {"type": "text", "encoding": "utf-8"},
            "images_extracted": [],
        }

    def _extract_markdown(self, content: bytes) -> dict[str, Any]:
        """Extract text from markdown file."""
        text = content.decode("utf-8", errors="ignore")
        # Convert markdown to plain text (removes formatting but keeps content)
        html = markdown.markdown(text)
        # Simple HTML tag removal (basic, but works for most cases)
        import re
        plain_text = re.sub(r"<[^>]+>", "", html)

        return {
            "text": plain_text,
            "metadata": {"type": "markdown", "original_length": len(text)},
            "images_extracted": [],
        }

    def _extract_pdf(self, content: bytes) -> dict[str, Any]:
        """Extract text from PDF file, including images with OCR."""
        text_parts = []
        images_extracted = []
        metadata = {"type": "pdf", "pages": 0, "images_found": 0}

        # Use pdfplumber for better text extraction
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                metadata["pages"] = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text from page
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

                    # Extract images from page and run OCR
                    if hasattr(page, "images") and page.images:
                        metadata["images_found"] += len(page.images)
                        for img in page.images:
                            try:
                                # Extract image and run OCR
                                # Note: pdfplumber doesn't directly extract images,
                                # so we'll use PyPDF2 as fallback for image extraction
                                pass  # Will handle in PyPDF2 fallback
                            except Exception as e:
                                logger.warning(f"Failed to extract image from page {page_num}: {e}")

        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}. Trying PyPDF2...")
            # Fallback to PyPDF2
            pdf_reader = PdfReader(io.BytesIO(content))
            metadata["pages"] = len(pdf_reader.pages)

            for page_num, page in enumerate(pdf_reader.pages, 1):
                # Extract text
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

                # Try to extract images (if any)
                if "/XObject" in page.get("/Resources", {}):
                    xobjects = page["/Resources"]["/XObject"].get_object()
                    for obj_name in xobjects:
                        obj = xobjects[obj_name]
                        if obj.get("/Subtype") == "/Image":
                            try:
                                # Extract image data
                                img_data = obj.get_data()
                                # Run OCR on extracted image
                                ocr_text = self._ocr_image(img_data)
                                if ocr_text:
                                    images_extracted.append({
                                        "page": page_num,
                                        "text": ocr_text,
                                    })
                                    text_parts.append(f"\n[Image {page_num} text]: {ocr_text}\n")
                            except Exception as e:
                                logger.warning(f"Failed to process image on page {page_num}: {e}")

        full_text = "\n\n".join(text_parts)

        return {
            "text": full_text,
            "metadata": metadata,
            "images_extracted": images_extracted,
        }

    def _extract_docx(self, content: bytes) -> dict[str, Any]:
        """Extract text from DOCX file, including images with OCR."""
        text_parts = []
        images_extracted = []
        metadata = {"type": "docx", "images_found": 0}

        doc = Document(io.BytesIO(content))

        # Extract text from paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    text_parts.append(row_text)

        # Extract images and run OCR
        # Note: python-docx doesn't directly expose images, but we can access them via relationships
        try:
            # Access document relationships to find images
            if hasattr(doc, "part") and hasattr(doc.part, "related_parts"):
                for rel in doc.part.related_parts.values():
                    if hasattr(rel, "content_type") and rel.content_type.startswith("image/"):
                        try:
                            img_data = rel.blob
                            ocr_text = self._ocr_image(img_data)
                            if ocr_text:
                                metadata["images_found"] += 1
                                images_extracted.append({"text": ocr_text})
                                text_parts.append(f"\n[Image text]: {ocr_text}\n")
                        except Exception as e:
                            logger.warning(f"Failed to extract text from image in DOCX: {e}")
        except Exception as e:
            logger.debug(f"Could not extract images from DOCX: {e}")

        full_text = "\n\n".join(text_parts)

        return {
            "text": full_text,
            "metadata": metadata,
            "images_extracted": images_extracted,
        }

    def _extract_image(self, content: bytes) -> dict[str, Any]:
        """Extract text from image using OCR."""
        try:
            image = Image.open(io.BytesIO(content))
            ocr_text = self._ocr_image(content)

            return {
                "text": ocr_text,
                "metadata": {
                    "type": "image",
                    "format": image.format,
                    "size": image.size,
                },
                "images_extracted": [{"text": ocr_text}] if ocr_text else [],
            }
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")

    def _ocr_image(self, image_data: bytes) -> str:
        """
        Perform OCR on image data.

        Args:
            image_data: Image bytes

        Returns:
            Extracted text from image
        """
        try:
            image = Image.open(io.BytesIO(image_data))

            if self.use_easyocr:
                return self._ocr_easyocr(image)
            else:
                return self._ocr_tesseract(image)
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""

    def _ocr_tesseract(self, image: Image.Image) -> str:
        """Use pytesseract for OCR."""
        try:
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Run OCR
            text = pytesseract.image_to_string(image, lang="eng")
            return text.strip()
        except pytesseract.TesseractNotFoundError:
            logger.error(
                "Tesseract OCR not found. Please install Tesseract OCR. "
                "See backend/OCR_SETUP.md for instructions."
            )
            return ""
        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {e}")
            return ""

    def _ocr_easyocr(self, image: Image.Image) -> str:
        """Use EasyOCR for better accuracy."""
        try:
            reader = self._get_easyocr_reader()
            if reader is None:
                # Fallback to tesseract
                return self._ocr_tesseract(image)

            import numpy as np
            img_array = np.array(image)

            # EasyOCR expects numpy array
            results = reader.readtext(img_array)

            # Combine all detected text
            text_parts = [result[1] for result in results if result[2] > 0.5]  # Confidence threshold
            return "\n".join(text_parts)
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}, falling back to tesseract")
            return self._ocr_tesseract(image)

