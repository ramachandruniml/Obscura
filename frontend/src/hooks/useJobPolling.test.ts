import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../api";
import type { JobResource } from "../types";
import { useJobPolling } from "./useJobPolling";

const job = (over: Partial<JobResource>): JobResource =>
  ({
    job_id: "j",
    status: "pending",
    kind: "image",
    method: "blur",
    confidence: 0.5,
    input_name: "a.png",
    created_at: "",
    expires_at: "",
    result_url: null,
    error: null,
    stats: {},
    ...over,
  }) as JobResource;

afterEach(() => vi.restoreAllMocks());

describe("useJobPolling", () => {
  it("polls until the job completes, then stops", async () => {
    const getJob = vi
      .spyOn(api, "getJob")
      .mockResolvedValueOnce(job({ status: "pending" }))
      .mockResolvedValueOnce(job({ status: "processing" }))
      .mockResolvedValue(job({ status: "complete" }));

    const { result } = renderHook(() => useJobPolling("j1", 5));
    await waitFor(() => expect(result.current.job?.status).toBe("complete"));

    const settled = getJob.mock.calls.length;
    await new Promise((r) => setTimeout(r, 40));
    expect(getJob.mock.calls.length).toBe(settled);
  });

  it("surfaces a failed job through job.error", async () => {
    vi.spyOn(api, "getJob").mockResolvedValue(job({ status: "failed", error: "boom" }));
    const { result } = renderHook(() => useJobPolling("j2", 5));
    await waitFor(() => expect(result.current.job?.status).toBe("failed"));
    expect(result.current.job?.error).toBe("boom");
  });

  it("captures a request error", async () => {
    vi.spyOn(api, "getJob").mockRejectedValue(new Error("network down"));
    const { result } = renderHook(() => useJobPolling("j3", 5));
    await waitFor(() => expect(result.current.error).toBe("network down"));
  });

  it("does nothing without a job id", () => {
    const getJob = vi.spyOn(api, "getJob");
    renderHook(() => useJobPolling(null));
    expect(getJob).not.toHaveBeenCalled();
  });
});
