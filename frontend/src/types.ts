export type RedactMethod = "blur" | "pixelate" | "box";
export type JobStatus = "pending" | "processing" | "complete" | "failed";

export const REDACT_METHODS: RedactMethod[] = ["blur", "pixelate", "box"];

export interface JobResource {
  job_id: string;
  status: JobStatus;
  kind: "image" | "video" | null;
  method: string;
  confidence: number;
  input_name: string | null;
  created_at: string;
  expires_at: string;
  result_url: string | null;
  error: string | null;
  stats: Record<string, number | string>;
}

export interface RedactAccepted {
  job_id: string;
  status: JobStatus;
}
