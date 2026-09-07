import type { JobResource, RedactAccepted } from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";

async function detail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
  } catch {
    /* non-JSON error body */
  }
  return `${res.status} ${res.statusText}`.trim();
}

export async function postRedact(
  file: File,
  method: string,
  confidence: number,
): Promise<RedactAccepted> {
  const form = new FormData();
  form.append("file", file);
  form.append("method", method);
  form.append("confidence", String(confidence));

  const res = await fetch(`${BASE}/redact`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await detail(res));
  return (await res.json()) as RedactAccepted;
}

export async function getJob(id: string): Promise<JobResource> {
  const res = await fetch(`${BASE}/jobs/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(await detail(res));
  return (await res.json()) as JobResource;
}

export function resultUrl(id: string): string {
  return `${BASE}/jobs/${encodeURIComponent(id)}/result`;
}
