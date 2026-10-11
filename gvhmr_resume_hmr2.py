"""Resume a verified, contiguous HMR2 checkpoint partial download.

Use only when `gvhmr download` has stopped and left an `.incomplete` file.
The completed file is SHA-256 checked before it replaces the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url


SIZE = 2_709_494_041
SHA256 = "2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e"
NAME = "hmr2/epoch=10-step=25000.ckpt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("GVHMR-portable/inputs/checkpoints"))
    args = parser.parse_args()
    root = args.checkpoint_root.resolve()
    target = root / NAME
    if target.is_file() and target.stat().st_size == SIZE:
        print(f"Already complete: {target}")
        return
    cache = root / ".cache/huggingface/download/hmr2"
    partials = sorted(cache.glob(f"*.{SHA256}.*.incomplete"),
                      key=lambda file: file.stat().st_size, reverse=True)
    if not partials or partials[0].stat().st_size < 1_000_000:
        parser.error("No usable HMR2 partial file found; use the ordinary checkpoint downloader")
    part = partials[0]
    if not part.resolve().is_relative_to(root) or not target.resolve().is_relative_to(root):
        parser.error("Checkpoint paths must stay within the selected root")
    if part.stat().st_size > SIZE:
        parser.error("Partial file is larger than the published checkpoint")
    url = hf_hub_url("ryanrudes/gvhmr", NAME)
    print(f"Resuming {part.stat().st_size:,} / {SIZE:,} bytes")
    failures = 0
    while part.stat().st_size < SIZE:
        offset = part.stat().st_size
        try:
            with requests.get(url, headers={"Range": f"bytes={offset}-"},
                              stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                content_range = response.headers.get("Content-Range", "")
                if response.status_code != 206 or not content_range.startswith(f"bytes {offset}-"):
                    raise RuntimeError(f"Server did not honor resume range: {response.status_code} {content_range}")
                next_report = offset + 32 * 1024 * 1024
                with part.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            if handle.tell() >= next_report:
                                print(f"Downloaded {handle.tell():,} / {SIZE:,} bytes", flush=True)
                                next_report += 32 * 1024 * 1024
            failures = 0
        except requests.RequestException as exc:
            failures += 1
            if failures > 5:
                raise RuntimeError("Download failed repeatedly; partial file preserved") from exc
            print(f"Connection interrupted; retrying from {part.stat().st_size:,} bytes")
            time.sleep(5)
    if part.stat().st_size != SIZE:
        raise RuntimeError(f"Incorrect checkpoint size: {part.stat().st_size:,} != {SIZE:,}")
    print("Verifying SHA-256...")
    digest = hashlib.sha256()
    with part.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise RuntimeError(f"Checkpoint hash mismatch: {digest.hexdigest()}; partial file preserved")
    target.parent.mkdir(parents=True, exist_ok=True)
    part.replace(target)
    print(f"Verified checkpoint: {target}")


if __name__ == "__main__":
    main()
