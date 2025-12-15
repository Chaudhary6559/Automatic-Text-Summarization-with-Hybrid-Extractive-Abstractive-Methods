# Frontend Setup Guide

## Option 1: Fix npm Installation (Recommended)

Try these commands in order:

```bash
# 1. Clear npm cache
npm cache clean --force

# 2. Try installing again
cd frontend
npm install

# 3. If still failing, try with different registry
npm install --registry https://registry.npmjs.org/

# 4. If network is unstable, increase retries
npm install --fetch-retries=5 --fetch-retry-mintimeout=20000
```

If npm install succeeds, then run:
```bash
npm run dev
```

## Option 2: Use Standalone HTML (No npm required!)

If npm installation continues to fail, you can use the standalone HTML version:

1. **Make sure your backend is running:**
   ```bash
   python -m backend.main
   ```

2. **Open the standalone HTML file:**
   - Simply double-click `frontend/standalone.html`
   - Or open it in your browser: `file:///path/to/frontend/standalone.html`
   - Or use a simple HTTP server:
     ```bash
     cd frontend
     python -m http.server 8080
     # Then open http://localhost:8080/standalone.html
     ```

The standalone version has all the same features:
- ✅ Document summarization
- ✅ Configurable parameters (top-k, max/min length)
- ✅ Display of extractive sentences
- ✅ Evaluation metrics (ROUGE, BLEU, METEOR, BERTScore)
- ✅ Beautiful UI with statistics

**No npm, no build step, no dependencies!** Just open the HTML file and it works.

## Troubleshooting

### Backend Connection Error
If you see "Failed to connect" errors:
- Make sure backend is running on `http://127.0.0.1:8000`
- Check the API_BASE URL in standalone.html (line ~300)
- Verify backend health: `http://127.0.0.1:8000/health`

### CORS Errors
If you see CORS errors when using standalone.html:
- Make sure you're opening it via HTTP server, not file://
- Or update backend CORS settings in `backend/app.py`

