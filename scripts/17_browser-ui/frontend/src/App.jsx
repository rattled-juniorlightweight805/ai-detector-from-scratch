import { useEffect, useMemo, useState } from "react";

const MIN_WORDS = 50;
const MIN_CHUNK_SIZE = 10;

function countWords(text) {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

function Result({ modelLabel, result }) {
  const score = Number(result.score);
  const aiLeaning = score >= 50;
  const accent = aiLeaning ? "#c5522d" : "#287a63";

  return (
    <section className="result" aria-live="polite">
      <div
        className="score-ring"
        style={{
          "--score": `${score * 3.6}deg`,
          "--accent": accent,
        }}
        aria-label={`${score.toFixed(1)} percent AI-generated score`}
      >
        <div className="score-center">
          <strong>{score.toFixed(1)}%</strong>
          <span>AI score</span>
        </div>
      </div>
      <div className="result-copy">
        <p className="eyebrow">{modelLabel} result</p>
        <h2>The classifier leans {result.label}.</h2>
        <p>
          {result.chunks
            ? "This is the token-weighted average of the chunk scores."
            : "This score reflects the model's output for the submitted text."} It
          is not proof of authorship.
        </p>
        <span>
          {result.word_count.toLocaleString()} words analyzed
          {result.chunks ? ` in ${result.chunks.length} chunks` : ""}
        </span>
      </div>
    </section>
  );
}

function ChunkAnalysis({ result }) {
  if (!result.chunks?.length) return null;

  return (
    <section className="chunk-analysis" aria-label="Chunk analysis">
      <div className="chunk-analysis-header">
        <div>
          <p className="eyebrow">Chunk analysis</p>
          <h2>{result.chunk_size} tokens per chunk</h2>
        </div>
        <div className="chunk-legend" aria-label="Chunk color legend">
          <span className="legend-ai">AI-leaning</span>
          <span className="legend-human">Human-leaning</span>
        </div>
      </div>
      <div className="highlighted-text">
        {result.chunks.map((chunk, index) => {
          const score = Number(chunk.score);
          const aiLeaning = score >= 50;
          const strength = aiLeaning ? score / 100 : 1 - score / 100;
          const background = aiLeaning
            ? `rgba(197, 82, 45, ${0.08 + strength * 0.16})`
            : `rgba(40, 122, 99, ${0.08 + strength * 0.16})`;
          return (
            <span
              className={aiLeaning ? "chunk-ai" : "chunk-human"}
              key={`${index}-${chunk.token_count}`}
              style={{ backgroundColor: background }}
              title={`Chunk ${index + 1}: ${score.toFixed(1)}% AI score (${chunk.token_count} tokens)`}
            >
              {chunk.text}
            </span>
          );
        })}
      </div>
      <p className="chunk-note">
        Hover over a section to see its score. The final chunk may contain
        fewer tokens.
      </p>
    </section>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [modelLabel, setModelLabel] = useState("Loading model...");
  const [chunkSize, setChunkSize] = useState("");
  const [maxChunkSize, setMaxChunkSize] = useState(null);
  const wordCount = useMemo(() => countWords(text), [text]);
  const wordsRemaining = Math.max(0, MIN_WORDS - wordCount);
  const parsedChunkSize = chunkSize === "" ? null : Number(chunkSize);
  const chunkSizeIsValid = parsedChunkSize === null || (
    Number.isInteger(parsedChunkSize)
    && parsedChunkSize >= MIN_CHUNK_SIZE
    && (maxChunkSize === null || parsedChunkSize <= maxChunkSize)
  );
  const canSubmit = wordCount >= MIN_WORDS && chunkSizeIsValid && !loading;

  useEffect(() => {
    let cancelled = false;

    async function loadModelStatus() {
      try {
        const response = await fetch("/api/health");
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Model status is unavailable.");
        }
        if (!cancelled) {
          setModelLabel(payload.model_label || payload.model);
          const reportedLimit = Number(payload.max_chunk_size);
          setMaxChunkSize(
            Number.isFinite(reportedLimit) && reportedLimit > 0
              ? reportedLimit
              : null,
          );
        }
      } catch {
        if (!cancelled) {
          setModelLabel("Model unavailable");
        }
      }
    }

    loadModelStatus();
    return () => {
      cancelled = true;
    };
  }, []);

  async function checkText() {
    if (!canSubmit) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("/api/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, chunk_size: parsedChunkSize }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "The text could not be checked.");
      }
      setResult(payload);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The text could not be checked.",
      );
    } finally {
      setLoading(false);
    }
  }

  function clearText() {
    setText("");
    setResult(null);
    setError("");
  }

  return (
    <main className="page-shell">
      <header className="site-header">
        <a className="brand" href="/" aria-label="Local AI Detector home">
          <span className="brand-mark" aria-hidden="true">A</span>
          <span>Local AI Detector</span>
        </a>
        <span className="model-pill">{modelLabel}</span>
      </header>

      <div className="intro">
        <p className="eyebrow">Runs entirely on this computer</p>
        <h1>Check whether text looks AI-generated.</h1>
        <p>
          Paste at least 50 words. The local classifier returns a score from
          0 to 100.
        </p>
      </div>

      <section className="checker-card" aria-label="AI text checker">
        <div className="checker-toolbar">
          <label htmlFor="text-input">Text</label>
          <span>{wordCount.toLocaleString()} words</span>
        </div>

        <div className="text-area-wrap">
          <textarea
            id="text-input"
            value={text}
            onChange={(event) => {
              setText(event.target.value);
              setResult(null);
              setError("");
            }}
            placeholder="Paste text here to check for AI..."
            spellCheck="true"
          />
          {text && (
            <button className="clear-button" type="button" onClick={clearText}>
              Clear
            </button>
          )}
        </div>

        <div className="checker-actions">
          <div className="analysis-settings">
            <label className="chunk-control" htmlFor="chunk-size">
              <span>Chunk size <small>tokens</small></span>
              <input
                id="chunk-size"
                type="number"
                min={MIN_CHUNK_SIZE}
                max={maxChunkSize ?? undefined}
                step="1"
                value={chunkSize}
                onChange={(event) => {
                  setChunkSize(event.target.value);
                  setResult(null);
                  setError("");
                }}
                placeholder="Whole text"
              />
            </label>
            <div className="input-status" aria-live="polite">
              {wordsRemaining > 0
                ? `${wordsRemaining} more ${wordsRemaining === 1 ? "word" : "words"} required`
                : !chunkSizeIsValid
                  ? maxChunkSize === null
                    ? `Use at least ${MIN_CHUNK_SIZE} tokens`
                    : `Use ${MIN_CHUNK_SIZE} to ${maxChunkSize.toLocaleString()} tokens`
                  : parsedChunkSize === null
                    ? "The whole text will be analyzed once"
                    : `Non-overlapping ${parsedChunkSize}-token chunks`}
            </div>
          </div>
          <button
            className="check-button"
            type="button"
            disabled={!canSubmit}
            onClick={checkText}
          >
            {loading ? "Checking..." : "Check for AI"}
          </button>
        </div>

        {error && <p className="error-message" role="alert">{error}</p>}
      </section>

      {result && <Result modelLabel={modelLabel} result={result} />}
      {result && <ChunkAnalysis result={result} />}
    </main>
  );
}
