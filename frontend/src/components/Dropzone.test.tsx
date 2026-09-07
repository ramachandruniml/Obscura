import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Dropzone } from "./Dropzone";

function pick(file: File) {
  fireEvent.change(screen.getByLabelText("file"), { target: { files: [file] } });
}

describe("Dropzone", () => {
  it("accepts a supported file", () => {
    const onFile = vi.fn();
    render(<Dropzone file={null} onFile={onFile} />);
    const f = new File(["x"], "clip.mp4", { type: "video/mp4" });
    pick(f);
    expect(onFile).toHaveBeenCalledWith(f);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("rejects an unsupported type and reports it", () => {
    const onFile = vi.fn();
    render(<Dropzone file={null} onFile={onFile} />);
    pick(new File(["x"], "notes.txt", { type: "text/plain" }));
    expect(onFile).toHaveBeenLastCalledWith(null);
    expect(screen.getByRole("alert")).toHaveTextContent(/unsupported file type/i);
  });

  it("rejects an oversized file", () => {
    const onFile = vi.fn();
    render(<Dropzone file={null} onFile={onFile} />);
    const f = new File(["x"], "big.png", { type: "image/png" });
    Object.defineProperty(f, "size", { value: 200 * 1024 * 1024 });
    pick(f);
    expect(onFile).toHaveBeenLastCalledWith(null);
    expect(screen.getByRole("alert")).toHaveTextContent(/max 100 MB/i);
  });

  it("shows the selected file name", () => {
    render(
      <Dropzone file={new File(["x"], "me.png", { type: "image/png" })} onFile={vi.fn()} />,
    );
    expect(screen.getByText(/me\.png/)).toBeInTheDocument();
  });
});
