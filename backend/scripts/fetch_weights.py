"""Download detector weights into backend/weights/.

Usage (from backend/):
    uv run python scripts/fetch_weights.py                 # everything it can
    uv run python scripts/fetch_weights.py --only yolov8face
    uv run python scripts/fetch_weights.py --retinaface-url https://.../mnet.pth
    uv run python scripts/fetch_weights.py --retinaface-gdrive-id <id>   # needs `gdown`

Both backends auto-download from public mirrors by default:
    YOLOv8-face  -> akanametov/yolo-face GitHub release   (override: --yolov8-url)
    RetinaFace   -> py-feat/retinaface on HuggingFace     (override: --retinaface-url
                    or --retinaface-gdrive-id for the upstream Google Drive copy)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

YOLOV8_FACE_DEFAULT_URL = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)
YOLOV8_FACE_FILENAME = "yolov8n-face.pt"

RETINAFACE_FILENAME = "retinaface_mobilenet0.25.pth"
# biubug6's checkpoint, re-hosted (biubug6-format state dict). Override with --retinaface-url.
RETINAFACE_DEFAULT_URL = (
    "https://huggingface.co/py-feat/retinaface/resolve/main/mobilenet0.25_Final.pth"
)
RETINAFACE_GDRIVE_HINT = (
    "biubug6/Pytorch_Retinaface README -> 'mobilenet0.25' checkpoint (Google Drive). Save it as:"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as fh:  # noqa: S310
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while chunk := resp.read(1 << 16):
            fh.write(chunk)
            read += len(chunk)
            if total:
                pct = 100 * read / total
                print(f"\r  {read / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct:4.1f}%)", end="")
    print()
    tmp.replace(dest)
    print(f"  saved {dest}  sha256={_sha256(dest)[:16]}…")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_yolov8(url: str) -> bool:
    dest = WEIGHTS_DIR / YOLOV8_FACE_FILENAME
    if dest.exists():
        print(f"[yolov8face] already present: {dest}")
        return True
    print("[yolov8face] downloading…")
    try:
        _download(url, dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[yolov8face] FAILED: {exc}")
        print(f"            download manually and save as: {dest}")
        return False


def fetch_retinaface(url: str | None, gdrive_id: str | None) -> bool:
    dest = WEIGHTS_DIR / RETINAFACE_FILENAME
    if dest.exists():
        print(f"[retinaface] already present: {dest}")
        return True

    if gdrive_id:
        try:
            import gdown  # type: ignore[import-untyped]
        except ImportError:
            print("[retinaface] --retinaface-gdrive-id needs `gdown` (uv pip install gdown)")
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=gdrive_id, output=str(dest), quiet=False)
        return dest.exists()

    print("[retinaface] downloading…")
    try:
        _download(url or RETINAFACE_DEFAULT_URL, dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[retinaface] FAILED: {exc}")

    print(f"            {RETINAFACE_GDRIVE_HINT}")
    print(f"            {dest}")
    print("            or re-run with --retinaface-url / --retinaface-gdrive-id")
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["retinaface", "yolov8face"], help="fetch just one backend")
    ap.add_argument("--yolov8-url", default=YOLOV8_FACE_DEFAULT_URL)
    ap.add_argument("--retinaface-url", default=None)
    ap.add_argument("--retinaface-gdrive-id", default=None)
    args = ap.parse_args(argv)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    if args.only in (None, "yolov8face"):
        ok &= fetch_yolov8(args.yolov8_url)
    if args.only in (None, "retinaface"):
        ok &= fetch_retinaface(args.retinaface_url, args.retinaface_gdrive_id)

    print("\ndone." if ok else "\nsome weights are missing — see messages above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
