import type { JobStatus } from "../types";

const LABEL: Record<JobStatus, string> = {
  pending: "Queued",
  processing: "Processing",
  complete: "Complete",
  failed: "Failed",
};

const STYLE: Record<JobStatus, string> = {
  pending: "bg-slate-700 text-slate-200",
  processing: "bg-sky-900 text-sky-200",
  complete: "bg-emerald-900 text-emerald-200",
  failed: "bg-rose-900 text-rose-200",
};

export function JobProgress({ status }: { status: JobStatus }) {
  const busy = status === "pending" || status === "processing";
  return (
    <div className="flex items-center gap-2">
      {busy && (
        <span
          data-testid="spinner"
          className="h-3 w-3 animate-spin rounded-full border-2 border-sky-400 border-t-transparent"
        />
      )}
      <span className={`rounded-full px-2.5 py-0.5 text-sm ${STYLE[status]}`}>{LABEL[status]}</span>
    </div>
  );
}
