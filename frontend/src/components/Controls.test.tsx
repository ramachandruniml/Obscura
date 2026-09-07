import { fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";
import type { RedactMethod } from "../types";
import { Controls } from "./Controls";

function setup(props: Partial<ComponentProps<typeof Controls>> = {}) {
  const onMethod = vi.fn();
  const onConfidence = vi.fn();
  render(
    <Controls
      method="blur"
      confidence={0.5}
      onMethod={onMethod}
      onConfidence={onConfidence}
      {...props}
    />,
  );
  return { onMethod, onConfidence };
}

describe("Controls", () => {
  it("lists the three redaction methods", () => {
    setup();
    expect(screen.getAllByRole("option").map((o) => o.textContent)).toEqual([
      "blur",
      "pixelate",
      "box",
    ]);
  });

  it("emits method changes", () => {
    const { onMethod } = setup();
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "pixelate" } });
    expect(onMethod).toHaveBeenCalledWith<[RedactMethod]>("pixelate");
  });

  it("shows the confidence value and emits changes", () => {
    const { onConfidence } = setup();
    expect(screen.getByText("0.50")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("confidence threshold"), { target: { value: "0.8" } });
    expect(onConfidence).toHaveBeenCalledWith(0.8);
  });

  it("disables both inputs when busy", () => {
    setup({ disabled: true });
    expect(screen.getByRole("combobox")).toBeDisabled();
    expect(screen.getByLabelText("confidence threshold")).toBeDisabled();
  });
});
