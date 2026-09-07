import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("./api", () => ({
  postRedact: vi.fn(),
  getJob: vi.fn(),
  resultUrl: (id: string) => `/api/jobs/${id}/result`,
}));

import * as api from "./api";
import { App } from "./App";

const mockApi = vi.mocked(api);

afterEach(() => vi.clearAllMocks());

const png = () => new File(["x"], "face.png", { type: "image/png" });

function chooseFileAndSubmit() {
  render(<App />);
  fireEvent.change(screen.getByLabelText("file"), { target: { files: [png()] } });
  fireEvent.click(screen.getByRole("button", { name: /redact faces/i }));
}

describe("App", () => {
  it("renders the title", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /obscura/i })).toBeInTheDocument();
  });

  it("runs a job to completion and shows the download link + face count", async () => {
    mockApi.postRedact.mockResolvedValue({ job_id: "job1", status: "pending" });
    mockApi.getJob
      .mockResolvedValueOnce({ job_id: "job1", status: "processing", stats: {} } as never)
      .mockResolvedValue({
        job_id: "job1",
        status: "complete",
        stats: { n_faces: 2 },
        result_url: "/api/jobs/job1/result",
      } as never);

    chooseFileAndSubmit();

    await waitFor(() => expect(mockApi.postRedact).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByText(/complete/i)).toBeInTheDocument(), {
      timeout: 3000,
    });

    expect(screen.getByRole("link", { name: /download/i })).toHaveAttribute(
      "href",
      "/api/jobs/job1/result",
    );
    expect(screen.getByText(/2 face\(s\) redacted/i)).toBeInTheDocument();
  });

  it("shows an error banner when the upload is rejected", async () => {
    mockApi.postRedact.mockRejectedValue(new Error("unsupported type text/plain"));
    chooseFileAndSubmit();
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(/unsupported type/i),
    );
  });

  it("shows failure status when the job fails", async () => {
    mockApi.postRedact.mockResolvedValue({ job_id: "job2", status: "pending" });
    mockApi.getJob.mockResolvedValue({
      job_id: "job2",
      status: "failed",
      error: "corrupt file",
      stats: {},
    } as never);

    chooseFileAndSubmit();
    await waitFor(() => expect(screen.getByText(/failed/i)).toBeInTheDocument());
    expect(screen.getByRole("alert")).toHaveTextContent(/corrupt file/i);
  });
});
