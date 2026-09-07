import { useMemo, useState } from "react";
import { postRedact, resultUrl } from "./api";
import { BeforeAfter } from "./components/BeforeAfter";
import { Controls } from "./components/Controls";
import { Dropzone } from "./components/Dropzone";
import { DownloadButton } from "./components/DownloadButton";
import { JobProgress } from "./components/JobProgress";
import { useJobPolling } from "./hooks/useJobPolling";
import type { RedactMethod } from "./types";

type Phase = "idle" | "submitting" | "tracking";

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
    phase === "submitting" ||
    (!!job && (job.status === "pending" || job.status === "processing"));
  const done = job?.status === "complete";

  const kind: "image" | "video" = useMemo(
    () => (file?.type.startsWith("video/") ? "video" : "image"),
    [file],
  );

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

  const faces = job?.stats?.n_faces;

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-3xl px-4 py-10">
        <header className="mb-8">
          <h1 className="text-3xl font-semibold tracking-tight">Obscura</h1>
          <p className="text-slate-400">
            Blur, pixelate, or box out every face before you share footage. No recognition,
            no retention.
          </p>
        </header>

        <section className="space-y-5 rounded-xl border border-slate-800 bg-slate-900/50 p-5">
          <Dropzone file={file} onFile={setFile} disabled={busy} />
          <Controls
            method={method}
            confidence={confidence}
            onMethod={setMethod}
            onConfidence={setConfidence}
            disabled={busy}
          />

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => void submit()}
              disabled={!file || busy}
              className="rounded-md bg-sky-600 px-4 py-2 font-medium text-white hover:bg-sky-500 disabled:opacity-40"
            >
              {busy ? "Working…" : "Redact faces"}
            </button>
            {job && <JobProgress status={job.status} />}
            {(done || error) && (
              <button
                type="button"
                onClick={reset}
                className="text-sm text-slate-400 underline hover:text-slate-200"
              >
                Start over
              </button>
            )}
          </div>

          {error && (
            <p role="alert" className="rounded-md bg-rose-950/60 px-3 py-2 text-sm text-rose-300">
              {error}
            </p>
          )}
        </section>

        {file && done && jobId && (
          <section className="mt-8 space-y-4">
            <BeforeAfter file={file} resultSrc={resultUrl(jobId)} kind={kind} />
            {typeof faces !== "undefined" && (
              <p className="text-sm text-slate-400">{faces} face(s) redacted.</p>
            )}
            <DownloadButton
              href={resultUrl(jobId)}
              filename={`${file.name.replace(/\.[^.]+$/, "")}_redacted`}
            />
          </section>
        )}
      </div>
    </main>
  );
}
