# Deployment Checklist

## 1. Backend (FastAPI)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm

uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 2
```

### Environment Variables

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `APP_HOST` | Bind address for FastAPI | `0.0.0.0` |
| `APP_PORT` | API port | `8000` |
| `VITE_API_BASE_URL` (frontend) | Base URL the SPA uses to reach the API | `http://localhost:8000` |

*(`APP_*` can be overridden by editing `backend/config/config.yaml`.)*

## 2. Frontend (React + Vite)

```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run build
npm run preview
```

Serve the contents of `frontend/dist` via any static web server (Netlify, Vercel, Nginx, etc.). The SPA communicates with the backend through the URL stored in `VITE_API_BASE_URL`.

## 3. Containerization (Optional)

1. Build backend image with Python base (e.g., `python:3.11-slim`), install requirements, expose port 8000.
2. Build frontend image using `node:20` for the build step and `nginx:alpine` for serving static assets.
3. Use Docker Compose or Kubernetes to network the two services and inject the `VITE_API_BASE_URL` into the frontend container at build time.

## 4. Observability

- Enable FastAPI logging levels via `config.yaml`.
- Wrap the frontend behind HTTPS and configure CORS origins accordingly in `backend/app.py`.
- Use GPU-backed infrastructure for production inference to keep BART latency within the target 2–5 seconds per document.


