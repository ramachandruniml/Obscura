import { afterEach, describe, expect, it, vi } from "vitest";
import { getJob, postRedact, resultUrl } from "./api";

const jsonOk = (body: unknown) =>
  ({ ok: true, status: 200, statusText: "OK", json: async () => body }) as Response;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("postRedact", () => {
  it("posts a multipart form with method + confidence and returns the job id", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonOk({ job_id: "abc", status: "pending" }));
    vi.stubGlobal("fetch", fetchMock);

    const file = new File(["x"], "a.png", { type: "image/png" });
    const out = await postRedact(file, "box", 0.4);

    expect(out).toEqual({ job_id: "abc", status: "pending" });
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/redact");
    expect(init.method).toBe("POST");
    const form = init.body as FormData;
    expect(form.get("method")).toBe("box");
    expect(form.get("confidence")).toBe("0.4");
    expect((form.get("file") as File).name).toBe("a.png");
  });

  it("throws the server-provided detail on an error response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 415,
        statusText: "Unsupported Media Type",
        json: async () => ({ detail: "unsupported type text/plain" }),
      }),
    );
    await expect(postRedact(new File([""], "a.txt"), "blur", 0.5)).rejects.toThrow(
      "unsupported type text/plain",
    );
  });

  it("falls back to status text when the error body is not JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        json: async () => {
          throw new Error("not json");
        },
      }),
    );
    await expect(postRedact(new File([""], "a.png", { type: "image/png" }), "blur", 0.5)).rejects.toThrow(
      "500 Internal Server Error",
    );
  });
});

describe("getJob", () => {
  it("parses the job resource", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonOk({ job_id: "j", status: "complete" })));
    expect((await getJob("j")).status).toBe("complete");
  });
});

it("resultUrl targets the result endpoint", () => {
  expect(resultUrl("j1")).toMatch(/\/jobs\/j1\/result$/);
});
