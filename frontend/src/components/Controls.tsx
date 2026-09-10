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
    <div className="controls">
      <label className="label">
        <span>Redaction method</span>
        <select
          className="field"
          value={method}
          disabled={disabled}
          onChange={(e) => onMethod(e.target.value as RedactMethod)}
        >
          {REDACT_METHODS.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </label>

      <label className="label">
        <span>
          Confidence threshold — <b className="mono">{confidence.toFixed(2)}</b>
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
        />
      </label>
    </div>
  );
}
