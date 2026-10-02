# OCR Setup Guide

The file extraction module supports OCR (Optical Character Recognition) for extracting text from images. This requires additional setup.

## Option 1: Tesseract OCR (Recommended - Faster)

### Windows:
1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install it (default location: `C:\Program Files\Tesseract-OCR`)
3. Add to PATH or set environment variable:
   ```powershell
   $env:TESSDATA_PREFIX = "C:\Program Files\Tesseract-OCR\tessdata"
   ```
4. Install Python package:
   ```bash
   pip install pytesseract
   ```

### Linux:
```bash
sudo apt-get install tesseract-ocr
pip install pytesseract
```

### macOS:
```bash
brew install tesseract
pip install pytesseract
```

## Option 2: EasyOCR (Better accuracy, slower)

EasyOCR doesn't require external dependencies but downloads models on first use:

```bash
pip install easyocr
```

Note: First run will download ~500MB of models. Works on CPU and GPU.

## Configuration

The `FileExtractor` class can use either:
- `pytesseract` (faster, requires Tesseract installation)
- `easyocr` (better accuracy, no external dependencies)

Default is `pytesseract`. To use EasyOCR, set `use_easyocr=True` when initializing:

```python
extractor = FileExtractor(use_easyocr=True)
```

## Testing OCR

You can test OCR functionality:

```python
from backend.src.file_extractor import FileExtractor

extractor = FileExtractor()
with open('test_image.png', 'rb') as f:
    result = extractor.extract_text(f.read(), 'test.png')
    print(result['text'])
```

## Troubleshooting

### "TesseractNotFoundError"
- Make sure Tesseract is installed and in PATH
- On Windows, you may need to set the path explicitly:
  ```python
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  ```

### "EasyOCR model download fails"
- Check internet connection
- Models are cached in `~/.EasyOCR/model/`
- You can manually download models if needed

### Poor OCR accuracy
- Use EasyOCR for better results (slower)
- Ensure images are clear and high resolution
- Pre-process images (contrast, brightness) if needed

