# File Upload & OCR Features

## Overview

The summarization system now supports multiple file formats with automatic text extraction and OCR capabilities for images.

## Supported File Formats

### Documents
- **PDF** (.pdf) - Extracts text from all pages, including OCR for embedded images
- **Word Documents** (.docx, .doc) - Extracts text from paragraphs, tables, and images
- **Text Files** (.txt) - Plain text extraction
- **Markdown** (.md) - Converts markdown to plain text

### Images (with OCR)
- PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP
- Text is extracted using OCR (Optical Character Recognition)
- Supports both Tesseract OCR and EasyOCR

## New API Endpoints

### 1. `/extract-text` (POST)
Extract text from uploaded files without summarization.

**Request:**
- `file`: Multipart file upload

**Response:**
```json
{
  "extracted_text": "Full extracted text...",
  "metadata": {
    "type": "pdf",
    "pages": 5,
    "images_found": 3
  },
  "images_extracted": [
    {"page": 1, "text": "Text from image 1"},
    {"page": 2, "text": "Text from image 2"}
  ],
  "filename": "document.pdf"
}
```

### 2. `/summarize-file` (POST)
Extract text from file and summarize in one step.

**Request:**
- `file`: Multipart file upload
- `top_k`: (optional) Number of sentences to extract
- `max_length`: (optional) Max summary length
- `min_length`: (optional) Min summary length
- `run_postprocess`: (optional) Apply post-processing
- `return_extractive_sentences`: (optional) Return extracted sentences

**Response:**
Same as `/summarize` endpoint

## Frontend Features

### Dual Input Modes

1. **Text Paste Mode** (Default)
   - Paste text directly into textarea
   - Works as before

2. **File Upload Mode**
   - Click to upload or drag-and-drop
   - Supports all file formats
   - Shows file info (name, size)
   - Automatic text extraction and summarization

### File Upload UI
- Drag-and-drop support
- File type validation
- File size display
- Progress indicators
- Error handling

## How It Works

### Text Extraction Pipeline

1. **File Type Detection**
   - Determines format from file extension
   - Routes to appropriate extractor

2. **Document Processing**
   - **PDF**: Uses pdfplumber for text, PyPDF2 for images
   - **DOCX**: Uses python-docx for text and images
   - **Images**: Direct OCR processing

3. **OCR Processing**
   - Extracts text from images in documents
   - Extracts text from standalone image files
   - Combines OCR text with document text
   - Context-aware integration

4. **Text Integration**
   - OCR text is merged with document text
   - Image text is marked with `[Image text]:` tags
   - Maintains document structure where possible

### Summarization Pipeline

1. **Extractive Phase**
   - Processes combined text (document + OCR)
   - Selects important sentences including those from images
   - Uses BERT embeddings + TextRank + features

2. **Abstractive Phase**
   - Refines extracted sentences
   - Generates fluent summary
   - Maintains context from images

3. **Post-Processing**
   - Removes repetition
   - Ensures coherence
   - Length control

## Usage Examples

### Python API

```python
import requests

# Upload and summarize a PDF
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    data = {
        'top_k': 5,
        'max_length': 150
    }
    response = requests.post(
        'http://localhost:8000/summarize-file',
        files=files,
        data=data
    )
    result = response.json()
    print(result['summary'])
```

### Frontend (Standalone HTML)

1. Open `frontend/standalone.html`
2. Click "Upload File" tab
3. Drag and drop or click to select file
4. Click "Generate Summary"
5. View results with extracted text from images

## OCR Setup

See `backend/OCR_SETUP.md` for detailed OCR installation instructions.

**Quick Setup:**
- **Windows**: Download Tesseract from GitHub
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

Or use EasyOCR (no external dependencies):
```bash
pip install easyocr
```

## Performance Notes

- **PDF Processing**: ~1-2 seconds per page
- **OCR Processing**: 
  - Tesseract: ~2-5 seconds per image
  - EasyOCR: ~5-10 seconds per image (better accuracy)
- **Large Files**: Files >10MB may take longer
- **Multiple Images**: Each image is processed sequentially

## Limitations

1. **Image Quality**: OCR accuracy depends on image quality
   - Clear, high-resolution images work best
   - Handwritten text may have lower accuracy
   - Complex layouts may need preprocessing

2. **File Size**: Very large files (>50MB) may timeout
   - Consider splitting large documents

3. **Language**: Currently optimized for English
   - Other languages may need additional OCR models

4. **Complex Formats**: 
   - Scanned PDFs work well
   - Complex tables may need manual review
   - Multi-column layouts may merge incorrectly

## Best Practices

1. **For Best OCR Results**:
   - Use high-resolution images (300+ DPI)
   - Ensure good contrast
   - Use EasyOCR for better accuracy (slower)

2. **For Large Documents**:
   - Consider extracting text first with `/extract-text`
   - Review extracted text before summarizing
   - Split very large documents

3. **For Images**:
   - Ensure text is clear and readable
   - Avoid heavily stylized fonts
   - Pre-process images if needed (contrast, brightness)

## Troubleshooting

### "Unsupported file format"
- Check file extension is in supported list
- Try renaming file with correct extension

### "OCR failed" or "No text extracted"
- Verify OCR is installed (see OCR_SETUP.md)
- Check image quality
- Try EasyOCR for better results

### "File too large"
- Split document into smaller parts
- Or use `/extract-text` first to check extraction

### "Timeout error"
- Large files or many images take time
- Consider processing in batches
- Check backend logs for details

