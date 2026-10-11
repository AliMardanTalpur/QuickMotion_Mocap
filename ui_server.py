r"""Local QuickMotion UI and sequential GVHMR job runner.

Run with ``.\.venv\Scripts\python.exe ui_server.py`` and open the printed URL.
The server binds to loopback only. Every processing stage is a separate process;
the GVHMR port runs its models sequentially on CPU for the 4 GB GPU machine.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
import zipfile
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from gvhmr_infer import missing_assets


ROOT = Path(__file__).resolve().parent
UI_ROOT = ROOT / "ui"
RUN_ROOT = ROOT / "runs" / "quickmotion"
GVHMR_ROOT = ROOT / "GVHMR-portable"
GVHMR_PYTHON = GVHMR_ROOT / ".venv" / "Scripts" / "python.exe"
MAX_UPLOAD_BYTES = 2 * 1024**3
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}
DEMO_ID = "demo-dance"
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
LOCK = threading.RLock()
JOBS: dict[str, dict] = {}
PENDING: queue.Queue[str] = queue.Queue()


class SingleInstanceHTTPServer(ThreadingHTTPServer):
    # ThreadingHTTPServer enables SO_REUSEADDR by default. On Windows that can
    # permit two servers to bind the same port and corrupt the same job state.
    allow_reuse_address = False

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_dir(job_id: str) -> Path:
    if not JOB_ID.fullmatch(job_id):
        raise ValueError("Invalid job ID")
    return RUN_ROOT / job_id


def write_manifest(job: dict) -> None:
    path = run_dir(job["id"]) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"manifest.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(job, indent=2), encoding="utf-8")
        for attempt in range(8):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temp.unlink(missing_ok=True)


def persist_runtime_job(job: dict) -> None:
    # A transient lock on Windows must not abandon a long-running child process.
    try:
        write_manifest(job)
    except OSError as exc:
        print(f"Warning: could not save job manifest: {exc}", file=sys.stderr)


def update_job(job_id: str, **changes: object) -> None:
    with LOCK:
        job = JOBS[job_id]
        job.update(changes)
        persist_runtime_job(job)


def append_log(job_id: str, line: str) -> None:
    line = line.strip()
    if not line:
        return
    with LOCK:
        job = JOBS[job_id]
        job["logs"] = (job.get("logs", []) + [line])[-120:]
        persist_runtime_job(job)


def load_jobs() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    for manifest in RUN_ROOT.glob("*/manifest.json"):
        if not JOB_ID.fullmatch(manifest.parent.name):
            continue
        try:
            job = json.loads(manifest.read_text(encoding="utf-8"))
            if job.get("id") != manifest.parent.name:
                continue
            if job.get("status") == "running":
                job["status"] = "interrupted"
                job["stage"] = "Interrupted by server restart"
                write_manifest(job)
            JOBS[job["id"]] = job
            if job.get("status") == "queued":
                PENDING.put(job["id"])
        except (OSError, ValueError, KeyError):
            continue


@lru_cache(maxsize=1)
def demo_video_metadata() -> tuple[float, float, int, int]:
    capture = cv2.VideoCapture(str(ROOT / "dance.mp4"))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return fps, count / fps if fps > 0 else 0.0, width, height
    finally:
        capture.release()


def demo_job() -> dict | None:
    output = ROOT / "runs" / "dance_gvhmr" / "dance_gvhmr_refined_v2.fbx"
    video = ROOT / "dance.mp4"
    if not (output.is_file() and video.is_file()):
        return None
    fps, duration, width, height = demo_video_metadata()
    return {
        "id": DEMO_ID,
        "name": "dance.mp4",
        "status": "complete",
        "stage": "Complete",
        "progress": 100,
        "created": datetime.fromtimestamp(output.stat().st_mtime, timezone.utc).isoformat(),
        "completed": datetime.fromtimestamp(output.stat().st_mtime, timezone.utc).isoformat(),
        "model": "GVHMR · single camera",
        "fps": round(fps, 3), "duration": round(duration, 3),
        "width": width, "height": height,
        "demo": True,
        "logs": ["Existing dance result imported from the project."],
    }


def public_jobs() -> list[dict]:
    with LOCK:
        items = [dict(job) for job in JOBS.values()]
    demo = demo_job()
    if demo:
        items.append(demo)
    return sorted(items, key=lambda item: item.get("created", ""), reverse=True)


def get_job(job_id: str) -> dict | None:
    if job_id == DEMO_ID:
        return demo_job()
    with LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def job_files(job: dict) -> dict[str, Path]:
    if job["id"] == DEMO_ID:
        base = ROOT / "runs" / "dance_gvhmr"
        return {
            "video": ROOT / "dance.mp4",
            "playback": ROOT / "dance.mp4",
            "fbx": base / "dance_gvhmr_refined_v2.fbx",
            "bvh": base / "dance_gvhmr_refined_v2.bvh",
            "points": base / "smplx24_refined_v2.npy",
            "preview": base / "motion_preview_refined_v2.jpg",
            "overlay": base / "review_overlay_refined_v2.jpg",
            "report": base / "dance_gvhmr_refined_v2.quality.json",
            "thumbnail": base / "quickmotion_thumbnail.jpg",
            "bundle": base / "quickmotion_bundle.zip",
        }
    base = run_dir(job["id"])
    return {
        "video": base / ("input" + job["extension"]),
        "playback": (base / "prepared.mp4" if (base / "prepared.mp4").is_file()
                     else base / ("input" + job["extension"])),
        "fbx": base / "motion.fbx",
        "bvh": base / "motion.bvh",
        "points": base / "smplx24_refined.npy",
        "preview": base / "motion_preview.jpg",
        "overlay": base / "review_overlay.jpg",
        "report": base / "motion.quality.json",
        "thumbnail": base / "thumbnail.jpg",
        "bundle": base / "quickmotion_bundle.zip",
    }


def make_thumbnail(video: Path, output: Path) -> tuple[float, float, int, int]:
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError("OpenCV could not open this video")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video FPS or dimensions are invalid")
        if max(width, height) > 1920 or min(width, height) > 1080 or fps > 60.5:
            raise ValueError("Upload a video up to 1080p and 60 FPS")
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(15, max(int(count) - 1, 0)))
        okay, frame = cap.read()
        if not okay:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            okay, frame = cap.read()
        if not okay:
            raise ValueError("This video has no readable frames")
        target_width = min(640, width)
        if width > target_width:
            frame = cv2.resize(frame, (target_width, round(height * target_width / width)))
        if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 86]):
            raise OSError("Could not write video thumbnail")
        return fps, count / fps if count > 0 else 0.0, width, height
    finally:
        cap.release()


def convert_to_mp4(source: Path, target: Path, job_id: str) -> None:
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Cannot decode source video for MP4 conversion")
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("OpenCV cannot create MP4 on this computer")
    frames = 0
    try:
        while True:
            okay, frame = cap.read()
            if not okay:
                break
            writer.write(frame)
            frames += 1
    finally:
        cap.release()
        writer.release()
    if frames == 0:
        raise ValueError("The source video has no decodable frames")
    append_log(job_id, f"Converted {frames} frames to MP4")


def run_step(job_id: str, stage: str, progress: int, command: list[str]) -> None:
    update_job(job_id, stage=stage, progress=progress)
    append_log(job_id, stage)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=flags,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            append_log(job_id, line)
        code = process.wait()
    finally:
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=False)
            else:
                process.terminate()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()
    if code != 0:
        raise RuntimeError(f"{stage} exited with code {code}. See the job log.")


def process_job(job_id: str) -> None:
    job = get_job(job_id)
    if not job or job["status"] != "queued":
        return
    paths = job_files(job)
    base = run_dir(job_id)
    update_job(job_id, status="running", stage="Checking models", progress=4,
               error=None)
    try:
        if not GVHMR_PYTHON.is_file():
            raise FileNotFoundError(f"GVHMR environment missing: {GVHMR_PYTHON}")
        missing = missing_assets(GVHMR_ROOT)
        if missing:
            raise FileNotFoundError("GVHMR assets missing: " + "; ".join(missing))
        video = paths["video"]
        if job["extension"] != ".mp4":
            update_job(job_id, stage="Preparing MP4", progress=8)
            video = base / "prepared.mp4"
            convert_to_mp4(paths["video"], video, job_id)
        infer = [str(GVHMR_PYTHON), "-u", str(ROOT / "gvhmr_infer.py"),
                 "--video", str(video), "--repo", str(GVHMR_ROOT),
                 "--output-dir", str(base)]
        if job.get("fast_preprocess", False):
            infer.append("--fast-preprocess")
        if job.get("quality_mode", True):
            infer.append("--flip-test")
        run_step(job_id, "Recovering 3D motion", 12, infer)
        main_python = sys.executable
        run_step(job_id, "Refining limbs", 78, [
            main_python, "-u", str(ROOT / "gvhmr_refine.py"),
            "--run-dir", str(base), "--output", str(paths["points"]),
        ])
        run_step(job_id, "Preparing previews", 85, [
            main_python, "-u", str(ROOT / "motion_preview.py"),
            "--input", str(paths["points"]), "--output", str(paths["preview"]),
            "--samples", "16",
        ])
        run_step(job_id, "Checking video alignment", 89, [
            main_python, "-u", str(ROOT / "gvhmr_review.py"),
            "--run-dir", str(base), "--joints", str(paths["points"]),
            "--output", str(paths["overlay"]), "--samples", "16",
        ])
        run_step(job_id, "Exporting FBX and BVH", 94, [
            main_python, "-u", str(ROOT / "stage3_export.py"),
            "--input", str(paths["points"]), "--fps", "30",
            "--bvh", str(paths["bvh"]), "--fbx", str(paths["fbx"]),
            "--ground", "fixed", "--foot-lock", "off",
        ])
        for kind in ("fbx", "bvh", "points", "preview", "overlay", "report"):
            if not paths[kind].is_file() or paths[kind].stat().st_size == 0:
                raise RuntimeError(f"Missing output: {kind}")
        update_job(job_id, status="complete", stage="Complete", progress=100,
                   completed=utc_now())
        append_log(job_id, "Ready to preview and download")
    except Exception as exc:
        append_log(job_id, f"ERROR: {exc}")
        update_job(job_id, status="failed", stage="Failed", error=str(exc))


def worker_loop() -> None:
    while True:
        job_id = PENDING.get()
        try:
            process_job(job_id)
        finally:
            PENDING.task_done()


def make_bundle(job: dict) -> Path:
    paths = job_files(job)
    bundle = paths["bundle"]
    if bundle.is_file():
        return bundle
    temp = bundle.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temp, "w", allowZip64=True) as archive:
        for kind in ("video", "fbx", "bvh", "preview", "overlay", "report"):
            path = paths[kind]
            if path.is_file():
                archive.write(path, f"QuickMotion/{path.name}",
                              compress_type=zipfile.ZIP_STORED)
    temp.replace(bundle)
    return bundle


class Handler(BaseHTTPRequestHandler):
    server_version = "QuickMotion/1.0"

    def log_message(self, format: str, *args: object) -> None:
        if self.command != "GET":
            super().log_message(format, *args)

    def send_json(self, data: object, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_error_json(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message}, status)

    def send_file(self, path: Path, download: bool = False) -> None:
        if not path.is_file():
            self.send_error_json("File not found", 404)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range", "")
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
        if match and size:
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else end
            elif match.group(2):
                start = max(0, size - int(match.group(2)))
            if start >= size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)
        partial = bool(match and size)
        self.send_response(206 if partial else 200)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix.lower() == ".fbx":
            mime = "application/octet-stream"
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        try:
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/system":
            missing = []
            if not GVHMR_PYTHON.is_file():
                missing.append("GVHMR Python environment")
            if not (GVHMR_ROOT / "gvhmr/inference/pipelines.py").is_file():
                missing.append("GVHMR checkout")
            if not missing:
                missing.extend(missing_assets(GVHMR_ROOT))
            self.send_json({"ready": not missing, "missing": missing,
                            "device": "CPU · sequential models",
                            "max_upload_bytes": MAX_UPLOAD_BYTES})
            return
        if path == "/api/jobs":
            self.send_json({"jobs": public_jobs()})
            return
        match = re.fullmatch(r"/api/jobs/([^/]+)(?:/(.*))?", path)
        if match:
            job = get_job(match.group(1))
            if not job:
                self.send_error_json("Asset not found", 404)
                return
            tail = match.group(2)
            if not tail:
                self.send_json({"job": job})
                return
            files = job_files(job)
            if tail == "points":
                source = files["points"]
                if not source.is_file():
                    self.send_error_json("3D points are not ready", 404)
                    return
                points = np.load(source, mmap_mode="r", allow_pickle=False)
                if points.ndim != 3 or points.shape[1:] != (24, 3):
                    self.send_error_json("Invalid 3D points", 500)
                    return
                step = max(1, math.ceil(len(points) / 600))
                sampled = np.asarray(points[::step], dtype=np.float32)
                x_origin = float(np.median(points[:, 0, 0]))
                depth_origin = float(np.median(points[:, 0, 2]))
                xyz = np.stack((sampled[..., 0] - x_origin,
                                sampled[..., 2] - depth_origin,
                                -sampled[..., 1]), axis=-1)
                floor = float(np.percentile(np.min(-points[:, [7, 8, 10, 11], 1],
                                                       axis=1), 5))
                xyz[..., 2] -= floor
                self.send_json({"fps": 30 / step, "frames": xyz.round(4).tolist()})
                return
            file_match = re.fullmatch(r"files/(video|playback|fbx|bvh|preview|overlay|report|thumbnail|bundle)", tail)
            if file_match:
                kind = file_match.group(1)
                if kind == "thumbnail" and not files[kind].is_file():
                    try:
                        make_thumbnail(files["video"], files[kind])
                    except (OSError, ValueError):
                        self.send_error_json("Thumbnail unavailable", 404)
                        return
                if kind == "bundle":
                    if job["status"] != "complete":
                        self.send_error_json("Asset is not complete", 409)
                        return
                    try:
                        make_bundle(job)
                    except OSError as exc:
                        self.send_error_json(str(exc), 500)
                        return
                self.send_file(files[kind], download=kind in {"fbx", "bvh", "bundle", "report"})
                return
        if path.startswith("/api/"):
            self.send_error_json("Endpoint not found", 404)
            return
        relative = path.lstrip("/") or "index.html"
        target = (UI_ROOT / relative).resolve()
        if target != UI_ROOT.resolve() and UI_ROOT.resolve() not in target.parents:
            self.send_error_json("Invalid path", 403)
            return
        if not target.is_file():
            target = UI_ROOT / "index.html"
        self.send_file(target)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/jobs":
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error_json("Invalid upload size")
                return
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                self.send_error_json("Video must be between 1 byte and 2 GB", 413)
                return
            name = Path(unquote(self.headers.get("X-Filename", "video.mp4"))).name
            extension = Path(name).suffix.lower()
            if extension not in ALLOWED_EXTENSIONS:
                self.send_error_json("Use an MP4, AVI, or MOV video")
                return
            job_id = uuid.uuid4().hex
            directory = run_dir(job_id)
            directory.mkdir(parents=True)
            video = directory / ("input" + extension)
            try:
                with video.open("wb") as output:
                    remaining = length
                    while remaining:
                        block = self.rfile.read(min(1024 * 1024, remaining))
                        if not block:
                            raise ValueError("Upload ended early")
                        output.write(block)
                        remaining -= len(block)
                fps, duration, width, height = make_thumbnail(video, directory / "thumbnail.jpg")
                quality = parse_qs(parsed.query).get("quality", ["1"])[0] != "0"
                fast = parse_qs(parsed.query).get("fast", ["0"])[0] == "1"
                job = {
                    "id": job_id, "name": name, "extension": extension,
                    "status": "queued", "stage": "Waiting to process", "progress": 2,
                    "created": utc_now(), "completed": None, "error": None,
                    "model": "GVHMR · fast CPU" if fast else "GVHMR · single camera",
                    "quality_mode": quality,
                    "fast_preprocess": fast,
                    "fps": round(fps, 3), "duration": round(duration, 3),
                    "width": width, "height": height, "logs": [],
                }
                with LOCK:
                    JOBS[job_id] = job
                    write_manifest(job)
                PENDING.put(job_id)
                self.send_json({"job": job}, 201)
            except (OSError, ValueError) as exc:
                shutil.rmtree(directory)
                self.send_error_json(str(exc))
            return
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/retry", parsed.path)
        if match:
            job_id = match.group(1)
            job = get_job(job_id)
            if not job:
                self.send_error_json("Asset not found", 404)
            elif job["status"] not in {"failed", "interrupted"}:
                self.send_error_json("Only failed or interrupted jobs can be retried", 409)
            else:
                update_job(job_id, status="queued", stage="Waiting to process",
                           progress=2, error=None, logs=[])
                PENDING.put(job_id)
                self.send_json({"job": get_job(job_id)})
            return
        self.send_error_json("Endpoint not found", 404)

    def do_DELETE(self) -> None:  # noqa: N802
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})", urlparse(self.path).path)
        if not match:
            self.send_error_json("Endpoint not found", 404)
            return
        job_id = match.group(1)
        with LOCK:
            job = JOBS.get(job_id)
            if not job:
                self.send_error_json("Asset not found", 404)
                return
            if job["status"] == "running":
                self.send_error_json("Wait for processing to finish before deleting", 409)
                return
            directory = run_dir(job_id).resolve()
            if directory.parent != RUN_ROOT.resolve():
                self.send_error_json("Invalid asset path", 403)
                return
            del JOBS[job_id]
        shutil.rmtree(directory)
        self.send_json({"deleted": job_id})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the browser")
    args = parser.parse_args()
    if not UI_ROOT.is_dir():
        parser.error(f"UI folder missing: {UI_ROOT}")
    url = f"http://127.0.0.1:{args.port}"
    # Also detect servers started with older versions, before the exclusive
    # socket option existed. A failed second launch must not touch manifests.
    try:
        with socket.create_connection(("127.0.0.1", args.port), timeout=1):
            print(f"QuickMotion is already running at {url}", flush=True)
            if args.open:
                webbrowser.open(url)
            return
    except (ConnectionRefusedError, TimeoutError):
        pass
    # Claim the port before touching saved jobs. A second UI launch must never
    # mark a job from the first server as interrupted or start duplicate work.
    try:
        server = SingleInstanceHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 10048 or exc.errno in {48, 98, 10048}:
            print(f"QuickMotion is already running at {url}", flush=True)
            if args.open:
                webbrowser.open(url)
            return
        raise
    load_jobs()
    threading.Thread(target=worker_loop, daemon=True, name="quickmotion-worker").start()
    print(f"QuickMotion UI: {url}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
