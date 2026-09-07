import { useRef, useState } from "react";

const ACCEPT = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "video/mp4",
  "video/quicktime",
  "video/webm",
];
const MAX_BYTES = 100 * 1024 * 1024;

interface Props {
  file: File | null;
  onFile: (file: File | null) => void;
  disabled?: boolean;
}

export function Dropzone({ file, onFile, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const consider = (candidate: File | undefined) => {
    if (!candidate) return;
    if (!ACCEPT.includes(candidate.type)) {
      setMessage(`Unsupported file type: ${candidate.type || "unknown"}`);
      onFile(null);
      return;
    }
    if (candidate.size > MAX_BYTES) {
      setMessage(`File is ${(candidate.size / 1e6).toFixed(1)} MB — max 100 MB.`);
      onFile(null);
      return;
    }
    setMessage(null);
    onFile(candidate);
  };

  return (
    <div>
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          consider(e.dataTransfer.files[0]);
        }}
        className={`w-full rounded-lg border-2 border-dashed p-8 text-center transition ${
          dragging ? "border-sky-400 bg-sky-950/40" : "border-slate-700 hover:border-slate-500"
        } disabled:opacity-50`}
      >
        {file ? (
          <span className="text-slate-200">
            {file.name} · {(file.size / 1e6).toFixed(1)} MB
          </span>
        ) : (
          <span className="text-slate-400">
            Drop an image or video here, or click to choose
          </span>
        )}
      </button>

      <input
        ref={inputRef}
        type="file"
        aria-label="file"
        accept={ACCEPT.join(",")}
        className="hidden"
        onChange={(e) => consider(e.target.files?.[0])}
      />

      {message && (
        <p className="mt-2 text-sm text-rose-400" role="alert">
          {message}
        </p>
      )}
    </div>
  );
}
