import { useEffect, useState } from "react";
import { getJob } from "../api";
import type { JobResource } from "../types";

export const POLL_INTERVAL_MS = 1500;
const MAX_WAIT_MS = 30 * 60 * 1000;

export interface PollingState {
  job: JobResource | null;
  error: string | null;
}

/** Poll GET /jobs/{id} until the job reaches a terminal state. */
export function useJobPolling(
  jobId: string | null,
  intervalMs: number = POLL_INTERVAL_MS,
): PollingState {
  const [job, setJob] = useState<JobResource | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setJob(null);
    setError(null);
    if (!jobId) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const startedAt = Date.now();

    const tick = async () => {
      try {
        const next = await getJob(jobId);
        if (cancelled) return;
        setJob(next);
        if (next.status === "complete" || next.status === "failed") return;
        if (Date.now() - startedAt > MAX_WAIT_MS) {
          setError("Timed out waiting for the job to finish.");
          return;
        }
        timer = setTimeout(() => void tick(), intervalMs);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    };

    void tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [jobId, intervalMs]);

  return { job, error };
}
