import { REDACT_METHODS, type RedactMethod } from "../types";

interface Props {
  method: RedactMethod;
  confidence: number;
  onMethod: (method: RedactMethod) => void;
  onConfidence: (confidence: number) => void;
  disabled?: boolean;
}

export function Controls({ method, confidence, onMethod, onConfidence, disabled }: Props) {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <label className="block">
        <span className="mb-1 block text-sm text-slate-400">Redaction method</span>
        <select
          value={method}
          disabled={disabled}
          onChange={(e) => onMethod(e.target.value as RedactMethod)}
          className="w-full rounded-md border border-slate-700 bg-slate-800 px-3 py-2 disabled:opacity-50"
        >
          {REDACT_METHODS.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </label>

      <label className="block">
        <span className="mb-1 block text-sm text-slate-400">
          Confidence threshold:{" "}
          <span className="tabular-nums text-slate-200">{confidence.toFixed(2)}</span>
        </span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={confidence}
          disabled={disabled}
          aria-label="confidence threshold"
          onChange={(e) => onConfidence(Number(e.target.value))}
          className="w-full accent-sky-500"
        />
      </label>
    </div>
  );
}
