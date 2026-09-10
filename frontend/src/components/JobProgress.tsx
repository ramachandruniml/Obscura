import type { JobStatus } from "../types";

const LABEL: Record<JobStatus, string> = {
  pending: "Queued",
  processing: "Processing",
  complete: "Complete",
  failed: "Failed",
};

export function JobProgress({ status }: { status: JobStatus }) {
  const busy = status === "pending" || status === "processing";
  return (
    <span className="progress">
      {busy && <span className="spinner" data-testid="spinner" />}
      <span className={`pill pill--${status}`}>{LABEL[status]}</span>
    </span>
  );
}
