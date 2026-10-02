import React, { useMemo, useState } from "react";

const defaultApiBase =
  (import.meta.env?.VITE_API_BASE_URL || "http://localhost:8000").replace(
    /\/$/,
    ""
  );

function App() {
  const apiBase = useMemo(() => defaultApiBase, []);

  const [textInput, setTextInput] = useState("");
  const [textSummary, setTextSummary] = useState("");
  const [textExtractive, setTextExtractive] = useState([]);

  const [file, setFile] = useState(null);
  const [fileSummary, setFileSummary] = useState("");
  const [fileExtractive, setFileExtractive] = useState([]);
  const [extractedText, setExtractedText] = useState("");
  const [extractionMetadata, setExtractionMetadata] = useState(null);
  const [imagesExtracted, setImagesExtracted] = useState([]);

  const [topK, setTopK] = useState("");
  const [maxLength, setMaxLength] = useState("");
  const [minLength, setMinLength] = useState("");
  const [runPostprocess, setRunPostprocess] = useState(true);
  const [returnExtractiveSentences, setReturnExtractiveSentences] =
    useState(true);

  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const resetMessages = () => {
    setMessage("");
    setError("");
  };

  const parseNumber = (value) => {
    if (value === "" || value === null || value === undefined) return undefined;
    const num = Number(value);
    return Number.isNaN(num) ? undefined : num;
  };

  const handleSummarizeText = async (event) => {
    event.preventDefault();
    resetMessages();

    if (!textInput || textInput.trim().length < 20) {
      setError("Please provide at least 20 characters to summarize.");
      return;
    }

    setLoading(true);
    try {
      const payload = {
        document: textInput,
        run_postprocess: runPostprocess,
        return_extractive_sentences: returnExtractiveSentences,
      };

      const topKValue = parseNumber(topK);
      const maxLenValue = parseNumber(maxLength);
      const minLenValue = parseNumber(minLength);

      if (topKValue !== undefined) payload.top_k = topKValue;
      if (maxLenValue !== undefined) payload.max_length = maxLenValue;
      if (minLenValue !== undefined) payload.min_length = minLenValue;

      const response = await fetch(`${apiBase}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || "Failed to summarize text.");
      }

      setTextSummary(data.summary || "");
      setTextExtractive(data.extractive_sentences || []);
      setMessage("Text summarized successfully.");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const buildFormDataWithOptions = () => {
    const formData = new FormData();
    formData.append("file", file);

    const topKValue = parseNumber(topK);
    const maxLenValue = parseNumber(maxLength);
    const minLenValue = parseNumber(minLength);

    if (topKValue !== undefined) formData.append("top_k", topKValue);
    if (maxLenValue !== undefined) formData.append("max_length", maxLenValue);
    if (minLenValue !== undefined) formData.append("min_length", minLenValue);

    formData.append("run_postprocess", runPostprocess);
    formData.append(
      "return_extractive_sentences",
      returnExtractiveSentences ? "true" : "false"
    );

    return formData;
  };

  const handleSummarizeFile = async (event) => {
    event.preventDefault();
    resetMessages();

    if (!file) {
      setError("Please choose a file first.");
      return;
    }

    setLoading(true);
    try {
      const formData = buildFormDataWithOptions();

      const response = await fetch(`${apiBase}/summarize-file`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || "Failed to summarize the file.");
      }

      setFileSummary(data.summary || "");
      setFileExtractive(data.extractive_sentences || []);
      setMessage("File summarized successfully.");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExtractTextOnly = async (event) => {
    event.preventDefault();
    resetMessages();

    if (!file) {
      setError("Please choose a file first.");
      return;
    }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${apiBase}/extract-text`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || "Failed to extract text.");
      }

      setExtractedText(data.extracted_text || "");
      setExtractionMetadata(data.metadata || null);
      setImagesExtracted(data.images_extracted || []);
      setMessage("Text extracted successfully.");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];
    setFile(selectedFile || null);
    setFileSummary("");
    setFileExtractive([]);
    setExtractedText("");
    setExtractionMetadata(null);
    setImagesExtracted([]);
    resetMessages();
  };

  return (
    <div className="page">
      <header className="header">
        <div>
          <p className="eyebrow">Hybrid Extractive + Abstractive</p>
          <h1>Summarizer & OCR</h1>
          <p className="lead">
            Paste text or upload files (PDF, DOCX, TXT, MD, images). The backend
            will extract with OCR when needed and generate a concise summary.
          </p>
          <p className="api-hint">
            API base: <code>{apiBase}</code>
          </p>
        </div>
      </header>

      <main className="grid">
        <section className="card">
          <div className="card-header">
            <h2>Summarize Text</h2>
            <p>Paste long-form text; we handle extraction and abstraction.</p>
          </div>
          <form className="stack" onSubmit={handleSummarizeText}>
            <label className="field">
              <span>Input text</span>
              <textarea
                rows={10}
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="Paste or type text here..."
              />
            </label>

            <OptionsForm
              topK={topK}
              maxLength={maxLength}
              minLength={minLength}
              runPostprocess={runPostprocess}
              returnExtractiveSentences={returnExtractiveSentences}
              onTopKChange={setTopK}
              onMaxLengthChange={setMaxLength}
              onMinLengthChange={setMinLength}
              onRunPostprocessChange={setRunPostprocess}
              onReturnExtractiveChange={setReturnExtractiveSentences}
            />

            <button type="submit" disabled={loading}>
              {loading ? "Processing..." : "Summarize Text"}
            </button>
          </form>

          <ResultPanel
            summary={textSummary}
            extractive={textExtractive}
            title="Text Summary"
          />
        </section>

        <section className="card">
          <div className="card-header">
            <h2>Upload File</h2>
            <p>
              Supports PDF, DOCX/DOC, TXT, Markdown, and images (PNG/JPG/GIF/BMP
              /TIFF/WebP) with OCR.
            </p>
          </div>

          <form className="stack" onSubmit={handleSummarizeFile}>
            <label className="field">
              <span>File</span>
              <input
                type="file"
                accept=".pdf,.doc,.docx,.txt,.md,.png,.jpg,.jpeg,.gif,.bmp,.tiff,.tif,.webp"
                onChange={handleFileChange}
              />
            </label>

            <OptionsForm
              topK={topK}
              maxLength={maxLength}
              minLength={minLength}
              runPostprocess={runPostprocess}
              returnExtractiveSentences={returnExtractiveSentences}
              onTopKChange={setTopK}
              onMaxLengthChange={setMaxLength}
              onMinLengthChange={setMinLength}
              onRunPostprocessChange={setRunPostprocess}
              onReturnExtractiveChange={setReturnExtractiveSentences}
            />

            <div className="actions">
              <button type="button" onClick={handleExtractTextOnly} disabled={loading}>
                {loading ? "Processing..." : "Extract Text Only"}
              </button>
              <button type="submit" disabled={loading}>
                {loading ? "Processing..." : "Summarize File"}
              </button>
            </div>
          </form>

          <ResultPanel
            summary={fileSummary}
            extractive={fileExtractive}
            title="File Summary"
          />

          {extractedText && (
            <div className="result-block">
              <h3>Extracted Text</h3>
              <pre className="pre">{extractedText}</pre>
            </div>
          )}

          {(extractionMetadata || imagesExtracted.length > 0) && (
            <div className="result-block meta">
              {extractionMetadata && (
                <div>
                  <h4>Metadata</h4>
                  <pre className="pre">
                    {JSON.stringify(extractionMetadata, null, 2)}
                  </pre>
                </div>
              )}
              {imagesExtracted.length > 0 && (
                <div>
                  <h4>OCR Snippets</h4>
                  <ul>
                    {imagesExtracted.map((img, idx) => (
                      <li key={idx}>
                        {img.page ? `Page ${img.page}: ` : ""}
                        {img.text || "(no text found)"}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </section>
      </main>

      {(message || error) && (
        <div className={`toast ${error ? "toast-error" : "toast-success"}`}>
          {error || message}
        </div>
      )}
    </div>
  );
}

function OptionsForm({
  topK,
  maxLength,
  minLength,
  runPostprocess,
  returnExtractiveSentences,
  onTopKChange,
  onMaxLengthChange,
  onMinLengthChange,
  onRunPostprocessChange,
  onReturnExtractiveChange,
}) {
  return (
    <div className="options">
      <label className="field small">
        <span>Top K (optional)</span>
        <input
          type="number"
          min="1"
          value={topK}
          onChange={(e) => onTopKChange(e.target.value)}
          placeholder="Auto"
        />
      </label>
      <label className="field small">
        <span>Max length</span>
        <input
          type="number"
          min="10"
          value={maxLength}
          onChange={(e) => onMaxLengthChange(e.target.value)}
          placeholder="Auto"
        />
      </label>
      <label className="field small">
        <span>Min length</span>
        <input
          type="number"
          min="10"
          value={minLength}
          onChange={(e) => onMinLengthChange(e.target.value)}
          placeholder="Auto"
        />
      </label>
      <label className="checkbox">
        <input
          type="checkbox"
          checked={runPostprocess}
          onChange={(e) => onRunPostprocessChange(e.target.checked)}
        />
        <span>Run post-process (deduplicate & smooth)</span>
      </label>
      <label className="checkbox">
        <input
          type="checkbox"
          checked={returnExtractiveSentences}
          onChange={(e) => onReturnExtractiveChange(e.target.checked)}
        />
        <span>Return extractive seed sentences</span>
      </label>
    </div>
  );
}

function ResultPanel({ summary, extractive, title }) {
  if (!summary && (!extractive || extractive.length === 0)) {
    return null;
  }

  return (
    <div className="result-block">
      <h3>{title}</h3>
      {summary && (
        <div>
          <h4>Summary</h4>
          <p className="summary">{summary}</p>
        </div>
      )}
      {extractive && extractive.length > 0 && (
        <div className="extractive">
          <h4>Extractive sentences</h4>
          <ol>
            {extractive.map((sentence, idx) => (
              <li key={idx}>{sentence}</li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}

export default App;

