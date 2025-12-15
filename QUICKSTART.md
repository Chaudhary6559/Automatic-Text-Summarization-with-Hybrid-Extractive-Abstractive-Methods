# Quick Start Guide

## Backend Setup

1. **Install dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

2. **Run the server:**
   ```bash
   python -m backend.main
   ```
   
   Or use the batch file:
   ```bash
   run_backend.bat
   ```

3. **Test the API:**
   - Open browser: `http://127.0.0.1:8000` - Should show API info
   - API Docs: `http://127.0.0.1:8000/docs` - Interactive Swagger UI
   - Health check: `http://127.0.0.1:8000/health`

## Frontend Setup

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Run the frontend:**
   ```bash
   npm run dev
   ```

3. **Access the app:**
   - Open browser: `http://localhost:5173`

## Testing the API Endpoints

### Using the Interactive Docs (Recommended)

1. Go to `http://127.0.0.1:8000/docs`
2. Click on `/summarize` endpoint
3. Click "Try it out"
4. Enter a JSON request like:
   ```json
   {
     "document": "Your long text document here...",
     "top_k": 5,
     "max_length": 150,
     "min_length": 60,
     "run_postprocess": true,
     "return_extractive_sentences": true
   }
   ```
5. Click "Execute"

### Using curl

```bash
# Health check
curl http://127.0.0.1:8000/health

# Summarize
curl -X POST "http://127.0.0.1:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "document": "Your document text here...",
    "top_k": 5,
    "max_length": 150
  }'
```

### Using Python

```python
import requests

# Summarize
response = requests.post(
    "http://127.0.0.1:8000/summarize",
    json={
        "document": "Your long document text here...",
        "top_k": 5,
        "max_length": 150,
        "min_length": 60,
        "run_postprocess": True,
        "return_extractive_sentences": True
    }
)
print(response.json())
```

## Troubleshooting

### "Not Found" error at root URL
- Make sure you're using the latest code with the root endpoint added
- Restart the server after code changes

### Model download issues
- First run will download large models (several GB)
- Ensure you have stable internet connection
- Models are cached in `./models/pretrained/`

### Import errors
- Make sure you're running from the project root directory
- Verify all dependencies are installed: `pip install -r backend/requirements.txt`

### Port already in use
- Change the port in `backend/config/config.yaml`
- Or set environment variable: `export PORT=8001`

