"""Shared HTTP download-with-resume helper, used by both the mcv and openslr sources."""
from pathlib import Path

import requests
from tqdm import tqdm

READ_TIMEOUT = 60  # seconds of silence on the socket before treating it as stalled
MAX_RETRIES = 8


def download_with_resume(url: str, archive_path: Path, headers: dict = None) -> Path:
    base_headers = dict(headers or {})
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        resume_pos = archive_path.stat().st_size if archive_path.exists() else 0
        req_headers = dict(base_headers)
        if resume_pos:
            req_headers["Range"] = f"bytes={resume_pos}-"
        try:
            with requests.get(
                url, stream=True, headers=req_headers, timeout=(10, READ_TIMEOUT),
            ) as r:
                if resume_pos and r.status_code == 416:
                    # Our local partial file doesn't line up with what the server has (e.g. the
                    # presigned URL rotated mid-download). Check the real size: if we already
                    # have it all, we're done; otherwise the partial file is stale, discard it.
                    head = requests.head(url, timeout=30)
                    true_size = int(head.headers.get("content-length", 0))
                    if true_size and resume_pos >= true_size:
                        print(f"{archive_path} is already fully downloaded ({resume_pos} bytes)")
                        return archive_path
                    print("Local partial file doesn't match the remote object, restarting download")
                    archive_path.unlink()
                    continue
                if resume_pos and r.status_code == 200:
                    # Server ignored the Range header, so it's sending the file from scratch.
                    resume_pos = 0
                r.raise_for_status()
                total = resume_pos + int(r.headers.get("content-length", 0))
                mode = "ab" if resume_pos else "wb"
                with open(archive_path, mode) as f, tqdm(
                    total=total, initial=resume_pos, unit="B", unit_scale=True, desc=archive_path.name,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            return archive_path
        except requests.exceptions.RequestException as e:
            print(f"Download stalled or failed (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                raise
            print("Retrying, resuming from the last downloaded byte...")

    raise RuntimeError("unreachable")
