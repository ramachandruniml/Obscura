interface Props {
  href: string;
  filename: string;
}

export function DownloadButton({ href, filename }: Props) {
  return (
    <a
      href={href}
      download={filename}
      className="inline-flex items-center rounded-md bg-emerald-600 px-4 py-2 font-medium text-white hover:bg-emerald-500"
    >
      Download redacted file
    </a>
  );
}
