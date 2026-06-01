#!/usr/bin/env python3
# Author: Tudor Mihaita
"""
Download and unpack the MuSiQue-Answerable dataset into data/musique/.

The dataset is hosted on Google Drive by the original authors.
Find the shareable link at: https://github.com/StonyBrookNLP/musique
then copy the file ID from the URL:
  https://drive.google.com/file/d/<FILE_ID>/view

Usage:
    # Download directly from Google Drive
    uv run python scripts/download_data.py --gdrive-id <FILE_ID>

    # If you already downloaded the zip manually
    uv run python scripts/download_data.py --zip path/to/musique.zip

The script is idempotent: it skips the download if the target file already exists.
"""
import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

import httpx
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragbench.config import settings

TARGET_FILE = "musique_ans_v1.0_dev.jsonl"
EXPECTED_LINES = 2417
_GDRIVE_BASE = "https://drive.usercontent.google.com/download"
_CHUNK = 1024 * 1024  # 1 MB


def _gdrive_url(file_id: str) -> str:
    return f"{_GDRIVE_BASE}?id={file_id}&export=download&confirm=t&authuser=0"


def download_from_gdrive(file_id: str, dest: Path) -> None:
    """Stream a public Google Drive file to `dest`."""
    url = _gdrive_url(file_id)
    logger.info(f"Downloading from Google Drive (id={file_id}) ...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=120, follow_redirects=True) as client:
        with client.stream("GET", url) as r:
            r.raise_for_status()
            content_type = r.headers.get("content-type", "")
            if "text/html" in content_type:
                body = r.read().decode(errors="replace")
                match = re.search(r'href="(/uc\?export=download[^"]+)"', body)
                if not match:
                    logger.error(
                        "Google Drive returned an HTML page instead of the file.\n"
                        "This usually means the file requires authentication or\n"
                        "the public download quota has been exceeded.\n"
                        "Download the zip manually and re-run with --zip <path>."
                    )
                    sys.exit(1)
                fallback_url = "https://drive.google.com" + match.group(1).replace("&amp;", "&")
                logger.info("Following Drive confirmation redirect ...")
                with client.stream("GET", fallback_url) as r2:
                    r2.raise_for_status()
                    _stream_to_file(r2, dest)
            else:
                _stream_to_file(r, dest)

    logger.success(f"Saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def _stream_to_file(response: httpx.Response, dest: Path) -> None:
    total = int(response.headers.get("content-length", 0))
    received = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_bytes(chunk_size=_CHUNK):
            f.write(chunk)
            received += len(chunk)
            if total:
                pct = received / total * 100
                print(f"\r  {received / 1e6:.1f} / {total / 1e6:.1f} MB  ({pct:.0f}%)", end="", flush=True)
    print()


def extract(zip_path: Path, dest_dir: Path) -> Path:
    """
    Unzip the MuSiQue archive and move TARGET_FILE to dest_dir.
    Returns the path to the extracted file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        matches = [m for m in members if m.endswith(TARGET_FILE)]
        if not matches:
            logger.error(
                f"{TARGET_FILE} not found in the zip archive.\n"
                f"Contents: {members[:20]}"
            )
            sys.exit(1)

        member = matches[0]
        logger.info(f"Extracting {member} ...")
        zf.extract(member, dest_dir)

        extracted = dest_dir / member
        final = dest_dir / TARGET_FILE
        if extracted != final:
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted), final)
            # Clean up any empty intermediate directories left by the move
            try:
                extracted.parent.rmdir()
            except OSError:
                pass

    return final


def verify(path: Path) -> None:
    logger.info(f"Verifying {path} ...")
    lines = sum(1 for _ in open(path))
    if lines != EXPECTED_LINES:
        logger.warning(
            f"Expected {EXPECTED_LINES} lines but found {lines}. "
            "The file may be incomplete or a different split."
        )
    else:
        logger.success(f"Verified: {lines} questions — looks correct.")


def main(args: argparse.Namespace) -> None:
    target = settings.musique_file  # data/musique/musique_ans_v1.0_dev.jsonl

    if target.exists():
        logger.info(f"{target} already exists — skipping download.")
        verify(target)
        return

    zip_path: Path | None = None
    tmp_zip = Path("data/musique/_musique_tmp.zip")

    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            logger.error(f"Zip file not found: {zip_path}")
            sys.exit(1)
    elif args.gdrive_id:
        download_from_gdrive(args.gdrive_id, tmp_zip)
        zip_path = tmp_zip
    else:
        logger.error(
            "Provide either --gdrive-id <ID> or --zip <path>.\n"
            "Get the Google Drive file ID from:\n"
            "  https://github.com/StonyBrookNLP/musique"
        )
        sys.exit(1)

    extracted = extract(zip_path, target.parent)

    if zip_path == tmp_zip:
        tmp_zip.unlink(missing_ok=True)

    verify(extracted)
    logger.success(f"Dataset ready at {extracted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--gdrive-id",
        metavar="ID",
        help="Google Drive file ID (from the share URL)",
    )
    src.add_argument(
        "--zip",
        metavar="PATH",
        help="Path to a locally downloaded MuSiQue zip archive",
    )
    main(parser.parse_args())
