import { useEffect, useState } from "react";

function Media({ src, kind }: { src: string; kind: "image" | "video" }) {
  const style = { display: "block", width: "100%" } as const;
  if (kind === "video") {
    return <video src={src} controls style={style} />;
  }
  return <img src={src} alt="" style={style} />;
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
    <div className="ba">
      <figure>
        <figcaption className="figcap">Before</figcaption>
        <div className="figure-frame">{beforeSrc && <Media src={beforeSrc} kind={kind} />}</div>
      </figure>
      <figure>
        <figcaption className="figcap">After</figcaption>
        <div className="figure-frame">
          <Media src={resultSrc} kind={kind} />
        </div>
      </figure>
    </div>
  );
}
