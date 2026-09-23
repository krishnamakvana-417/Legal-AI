"""
Google Drive PDF Fetcher
Downloads all PDF files from the configured Google Drive folder into ./data/judgments/
Supports public shared folders via gdown and service-account auth via google-api-python-client.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# pyrefly: ignore [missing-import]
import gdown
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config (overridable via env / caller) ────────────────────────────────────
DEFAULT_FOLDER_ID = "12pQVjSxHpY7AS8Zeiu375nq9rYRvwpXN"
DOWNLOAD_DIR = Path(os.getenv("JUDGMENTS_DIR", "./data/judgments"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Download directory: {path.resolve()}")


def _file_sha256(path: Path, prefix_len: int = 12) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:prefix_len]


# ── Download ──────────────────────────────────────────────────────────────────

def download_folder_gdown(folder_id: str, output_dir: Path) -> list[Path]:
    """
    Download all files from a public/shared Google Drive folder using gdown.
    Existing files are NOT re-downloaded (idempotent).
    """
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    logger.info(f"⬇️  Downloading from: {folder_url}")

    try:
        gdown.download_folder(
            url=folder_url,
            output=str(output_dir),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )
    except Exception as exc:
        logger.error(f"gdown download failed: {exc}")
        logger.warning("If the folder is private, authenticate via Google Drive API with a service account.")
        raise RuntimeError(f"Could not download folder {folder_id}: {exc}") from exc

    pdfs = sorted(output_dir.glob("**/*.pdf"))
    logger.info(f"✅ Found {len(pdfs)} PDF(s) after download.")
    return pdfs


def download_folder_api(folder_id: str, output_dir: Path, credentials_json: str) -> list[Path]:
    """
    Download files using google-api-python-client with a service account JSON.
    Use this for private/restricted Drive folders.

    Args:
        folder_id: Google Drive folder ID.
        output_dir: Local directory to save files.
        credentials_json: Path to service account JSON key file.
    """
    # pyrefly: ignore [missing-import]
    from google.oauth2 import service_account
    # pyrefly: ignore [missing-import]
    from googleapiclient.discovery import build
    # pyrefly: ignore [missing-import]
    from googleapiclient.http import MediaIoBaseDownload
    import io

    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_file(credentials_json, scopes=SCOPES)
    service = build("drive", "v3", credentials=creds)

    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
        fields="files(id, name, size)",
        pageSize=1000,
    ).execute()

    files = results.get("files", [])
    logger.info(f"Found {len(files)} PDFs via Drive API.")

    downloaded: list[Path] = []
    for file in tqdm(files, desc="Downloading PDFs"):
        dest = output_dir / file["name"]
        if dest.exists():
            logger.debug(f"Skipping (exists): {file['name']}")
            downloaded.append(dest)
            continue

        request = service.files().get_media(fileId=file["id"])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        dest.write_bytes(fh.getvalue())
        downloaded.append(dest)

    return downloaded


# ── Verification ──────────────────────────────────────────────────────────────

def verify_pdfs(pdf_paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """
    Verify each PDF is non-empty and parseable.
    Returns (valid_list, invalid_list).
    """
    # pyrefly: ignore [missing-import]
    import pypdf

    valid: list[Path] = []
    invalid: list[Path] = []

    for p in tqdm(pdf_paths, desc="🔍 Verifying PDFs", unit="file"):
        try:
            reader = pypdf.PdfReader(str(p))
            if len(reader.pages) > 0:
                valid.append(p)
            else:
                logger.warning(f"Empty PDF (0 pages): {p.name}")
                invalid.append(p)
        except Exception as exc:
            logger.warning(f"Unreadable PDF [{p.name}]: {exc}")
            invalid.append(p)

    logger.info(f"Verification: ✅ {len(valid)} valid  ❌ {len(invalid)} unreadable")
    return valid, invalid


# ── Manifest ─────────────────────────────────────────────────────────────────

def save_manifest(pdf_paths: list[Path], output_dir: Path) -> dict:
    """Save a JSON manifest of all verified PDFs."""
    manifest = {}
    for p in pdf_paths:
        manifest[p.name] = {
            "path": str(p.resolve()),
            "size_bytes": p.stat().st_size,
            "sha256_prefix": _file_sha256(p),
        }

    manifest_path = output_dir.parent / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"📋 Manifest saved → {manifest_path}  ({len(manifest)} entries)")
    return manifest


# ── Main Entry Point ──────────────────────────────────────────────────────────

def fetch_all(
    folder_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
    test_mode: bool = False,
    credentials_json: Optional[str] = None,
) -> list[Path]:
    """
    Download all PDFs from Google Drive and return verified paths.

    Args:
        folder_id: Google Drive folder ID (defaults to env / hardcoded default).
        output_dir: Local save directory (defaults to DOWNLOAD_DIR).
        test_mode: Skip download; scan existing files only.
        credentials_json: Optional service account JSON for private folders.

    Returns:
        List of verified, readable PDF Paths.
    """
    fid = folder_id or os.getenv("GOOGLE_DRIVE_FOLDER_ID", DEFAULT_FOLDER_ID)
    dest = output_dir or DOWNLOAD_DIR
    _ensure_dirs(dest)

    if test_mode:
        logger.info("🧪 TEST MODE — scanning existing files, skipping download.")
        pdfs = sorted(dest.glob("**/*.pdf"))
    elif credentials_json:
        pdfs = download_folder_api(fid, dest, credentials_json)
    else:
        pdfs = download_folder_gdown(fid, dest)

    valid, invalid = verify_pdfs(pdfs)

    if invalid:
        logger.warning("Skipping unreadable files:")
        for p in invalid:
            logger.warning(f"  ✗ {p.name}")

    save_manifest(valid, dest)
    return valid


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fetch PDFs from Google Drive")
    ap.add_argument("--folder-id", default=None, help="Override Google Drive folder ID")
    ap.add_argument("--output-dir", default=None, help="Override download directory")
    ap.add_argument("--test-mode", action="store_true", help="Skip download, scan existing files")
    ap.add_argument("--credentials", default=None, help="Path to service account JSON (for private folders)")
    args = ap.parse_args()

    paths = fetch_all(
        folder_id=args.folder_id,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        test_mode=args.test_mode,
        credentials_json=args.credentials,
    )

    print(f"\n{'='*50}")
    print(f"✅ Ready to process: {len(paths)} verified PDF(s)")
    print(f"{'='*50}")
    sys.exit(0 if paths else 1)
