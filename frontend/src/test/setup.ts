import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

// jsdom has no object-URL support; BeforeAfter needs it.
Object.defineProperty(URL, "createObjectURL", {
  writable: true,
  value: vi.fn(() => "blob:mock"),
});
Object.defineProperty(URL, "revokeObjectURL", {
  writable: true,
  value: vi.fn(),
});
