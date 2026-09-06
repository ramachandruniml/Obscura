/**
 * Skeleton shell. The real UI (upload dropzone, method selector, confidence
 * slider, job-status polling, before/after preview, download) lands in
 * Deliverable 8. Kept intentionally minimal until the API exists.
 */
export function App() {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-6">
      <div className="max-w-md text-center space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight">Obscura</h1>
        <p className="text-slate-400">
          Automatic face redaction for images and video. No recognition, no retention.
        </p>
        <p className="text-xs text-slate-600">
          Frontend skeleton — UI arrives in Deliverable 8.
        </p>
      </div>
    </main>
  );
}
