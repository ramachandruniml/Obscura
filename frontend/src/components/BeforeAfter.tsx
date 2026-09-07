import { useEffect, useState } from "react";

function Media({ src, kind }: { src: string; kind: "image" | "video" }) {
  if (kind === "video") {
    return <video src={src} controls className="w-full rounded-md bg-black" />;
  }
  return <img src={src} alt="" className="w-full rounded-md bg-black object-contain" />;
}

interface Props {
  file: File;
  resultSrc: string;
  kind: "image" | "video";
}

export function BeforeAfter({ file, resultSrc, kind }: Props) {
  const [beforeSrc, setBeforeSrc] = useState("");

  useEffect(() => {
    const url = URL.createObjectURL(file);
    setBeforeSrc(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <figure>
        <figcaption className="mb-1 text-sm text-slate-400">Before</figcaption>
        {beforeSrc && <Media src={beforeSrc} kind={kind} />}
      </figure>
      <figure>
        <figcaption className="mb-1 text-sm text-slate-400">After</figcaption>
        <Media src={resultSrc} kind={kind} />
      </figure>
    </div>
  );
}
