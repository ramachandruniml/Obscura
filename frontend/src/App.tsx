import { useMemo, useState } from "react";
import { postRedact, resultUrl } from "./api";
import { BeforeAfter } from "./components/BeforeAfter";
import { CoastScene } from "./components/CoastScene";
import { Controls } from "./components/Controls";
import { Dropzone } from "./components/Dropzone";
import { DownloadButton } from "./components/DownloadButton";
import { JobProgress } from "./components/JobProgress";
import { Arrow, BrandMark } from "./components/icons";
import { useJobPolling } from "./hooks/useJobPolling";
import type { RedactMethod } from "./types";

type Phase = "idle" | "submitting" | "tracking";

const STEPS = [
  { t: "Add a file", d: "Drop in a photo or a short video." },
  {
    t: "Choose a style",
    d: "Blur, pixelate, or a solid box — and set the detection threshold.",
  },
  { t: "Download", d: "Every face covered. Your file is deleted once it's done." },
];

export function App() {
  const [file, setFile] = useState<File | null>(null);
  const [method, setMethod] = useState<RedactMethod>("blur");
  const [confidence, setConfidence] = useState(0.5);
  const [jobId, setJobId] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [submitError, setSubmitError] = useState<string | null>(null);

  const { job, error: pollError } = useJobPolling(jobId);

  const error =
    submitError ?? pollError ?? (job?.status === "failed" ? (job.error ?? "Job failed.") : null);
  const busy =
    phase === "submitting" || (!!job && (job.status === "pending" || job.status === "processing"));
  const done = job?.status === "complete";

  const kind: "image" | "video" = useMemo(
    () => (file?.type.startsWith("video/") ? "video" : "image"),
    [file],
  );
  const faces = job?.stats?.n_faces;

  const submit = async () => {
    if (!file) return;
    setSubmitError(null);
    setPhase("submitting");
    try {
      const accepted = await postRedact(file, method, confidence);
      setJobId(accepted.job_id);
      setPhase("tracking");
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
      setPhase("idle");
    }
  };

  const reset = () => {
    setFile(null);
    setJobId(null);
    setPhase("idle");
    setSubmitError(null);
  };

  return (
    <div className="page">
      <header className="nav">
        <div className="wrap nav-inner">
          <a className="brand" href="#top">
            <BrandMark />
            <span>OBSCURA</span>
          </a>
          <nav className="nav-links">
            <a href="#how">How it works</a>
          </nav>
          <a className="btn nav-cta" href="#tool">
            Redact a file <Arrow />
          </a>
        </div>
      </header>

      <section className="hero" id="top">
        <div className="wrap hero-copy">
          <p className="eyebrow">Share the view — not the faces.</p>
          <h1 className="display">
            Blur every face.
            <br />
            Share with a clear conscience.
          </h1>
          <p className="subtle hero-sub">
            Automatic face redaction for photos and video. No recognition, no embeddings —
            your files are deleted once processing finishes.
          </p>
          <a className="btn hero-cta" href="#tool">
            Redact a file <Arrow />
          </a>
        </div>
        <CoastScene variant="hero" />
      </section>

      <section className="how wrap" id="how">
        <div className="how-strip">
          <CoastScene variant="strip" />
        </div>
        <h2 className="how-title">How it works</h2>
        <ol className="steps">
          {STEPS.map((s, i) => (
            <li className="step" key={s.t}>
              <span className="step-num">{i + 1}</span>
              <h3>{s.t}</h3>
              <p>{s.d}</p>
            </li>
          ))}
        </ol>
      </section>

      <main className="wrap tool" id="tool">
        <div className="card tool-card">
          <Dropzone file={file} onFile={setFile} disabled={busy} />
          <Controls
            method={method}
            confidence={confidence}
            onMethod={setMethod}
            onConfidence={setConfidence}
            disabled={busy}
          />

          <div className="tool-actions">
            <button
              type="button"
              className="btn"
              onClick={() => void submit()}
              disabled={!file || busy}
            >
              {busy ? "Working…" : "Redact faces"}
              {!busy && <Arrow />}
            </button>
            {job && <JobProgress status={job.status} />}
            {(done || error) && (
              <button type="button" className="link-ink" onClick={reset}>
                Start over
              </button>
            )}
          </div>

          {error && (
            <p role="alert" className="alert">
              {error}
            </p>
          )}
        </div>

        {file && done && jobId && (
          <div className="results">
            <BeforeAfter file={file} resultSrc={resultUrl(jobId)} kind={kind} />
            {typeof faces !== "undefined" && (
              <p className="subtle results-count">{faces} face(s) redacted.</p>
            )}
            <DownloadButton
              href={resultUrl(jobId)}
              filename={`${file.name.replace(/\.[^.]+$/, "")}_redacted`}
            />
          </div>
        )}
      </main>

      <footer className="foot">
        <div className="wrap">No recognition · no embeddings · no retention.</div>
      </footer>
    </div>
  );
}
