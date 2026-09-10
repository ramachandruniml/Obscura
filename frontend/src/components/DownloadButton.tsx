import { Arrow } from "./icons";

interface Props {
  href: string;
  filename: string;
}

export function DownloadButton({ href, filename }: Props) {
  return (
    <a className="btn" href={href} download={filename}>
      Download redacted file <Arrow />
    </a>
  );
}
