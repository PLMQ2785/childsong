from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import os
import json
import time
import uuid
import shutil
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
import re

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("yue-api")

# ---------------------------
# Config (env override)
# ---------------------------
YUE_PATH = os.environ.get("YUE_PATH", "/opt/YuE/inference")
CUDA_IDX = os.environ.get("CUDA_IDX", "0")
STAGE1_MODEL = os.environ.get("STAGE1_MODEL", "m-a-p/YuE-s1-7B-anneal-jp-kr-cot")
STAGE2_MODEL = os.environ.get("STAGE2_MODEL", "m-a-p/YuE-s2-1B-general")

JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/tmp/yue_jobs"))
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", "3600"))

MAX_GENRE_CHARS = int(os.environ.get("MAX_GENRE_CHARS", "2000"))
MAX_LYRICS_CHARS = int(os.environ.get("MAX_LYRICS_CHARS", "20000"))

INFER_TIMEOUT = int(os.environ.get("INFER_TIMEOUT", "1800"))          # 30 min
DEMUCS_TIMEOUT = int(os.environ.get("DEMUCS_TIMEOUT", "900"))         # 15 min
FFMPEG_TIMEOUT = int(os.environ.get("FFMPEG_TIMEOUT", "300"))         # 5 min

DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")
DEMUCS_DEVICE = os.environ.get("DEMUCS_DEVICE", "cuda")               # cuda/cpu

DEFAULT_VOCAL_SR = int(os.environ.get("DEFAULT_VOCAL_SR", "16000"))
DEFAULT_VOCAL_MONO = os.environ.get("DEFAULT_VOCAL_MONO", "1") == "1"

QUEUE_MAXSIZE = int(os.environ.get("QUEUE_MAXSIZE", "100"))
PROGRESS_WRITE_MIN_INTERVAL = float(os.environ.get("PROGRESS_WRITE_MIN_INTERVAL", "1.0"))

# Supported output extensions
OUTPUT_EXTS = tuple(x.strip() for x in os.environ.get(
    "OUTPUT_EXTS",
    ".wav,.mp3,.flac,.m4a,.ogg"
).split(",") if x.strip())

# CORS origins (comma separated)
CORS_ORIGINS = [x.strip() for x in os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5500,http://127.0.0.1:5500,"
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:8080,http://127.0.0.1:8080,"
    "http://localhost:10030, *"
).split(",") if x.strip()]

# ---------------------------
# App + in-process queue
# ---------------------------
app = FastAPI(
    title="YuE Music Generation API",
    version="2.2.0",
    description="Async job queue: YuE generate -> Demucs separate -> normalize vocals for lipsync",
)

# ---------------------------
# CORS (after app creation)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # 테스트 목적이면 False 권장
    allow_methods=["*"],
    allow_headers=["*"],
)

job_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
worker_task: Optional[asyncio.Task] = None

# ---------------------------
# Stage trace (stdout parsing)
# ---------------------------
_STAGE1_PCT_RE = re.compile(r"Stage1 inference\.\.\.:\s*(\d+)%")
_STAGE2_PCT_RE = re.compile(r"Stage\s*2 inference\.\.\.\s*(\d+)%")
_FRAC_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

# ---------------------------
# Request/Response models
# ---------------------------
class JobCreateRequest(BaseModel):
    genre: str
    lyrics: str

    run_n_segments: int = Field(2, ge=1, le=12)
    max_new_tokens: int = Field(3000, ge=128, le=10000)
    repetition_penalty: float = Field(1.1, ge=0.8, le=2.0)
    stage2_batch_size: int = Field(4, ge=1, le=32)

    separate_tracks: bool = Field(True)
    normalize_vocals: bool = Field(True)
    vocal_sr: int = Field(DEFAULT_VOCAL_SR, ge=8000, le=48000)
    vocal_mono: bool = Field(DEFAULT_VOCAL_MONO)

class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    status_url: str

# ---------------------------
# Helpers
# ---------------------------
def _now() -> float:
    return time.time()

def _ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    infer_py = Path(YUE_PATH) / "infer.py"
    if not infer_py.exists():
        raise RuntimeError(f"infer.py not found at {infer_py}")

def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id

def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"

def _input_path(job_id: str) -> Path:
    return _job_dir(job_id) / "input.json"

def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _set_status(job_id: str, **fields) -> None:
    sp = _status_path(job_id)
    base = _read_json(sp) if sp.exists() else {}

    # progress는 절대 감소하지 않게(단조 증가)
    if "progress" in fields:
        prev = int(base.get("progress", 0) or 0)
        fields["progress"] = max(prev, int(fields["progress"] or 0))

    base.update(fields)
    base["updated_at"] = _now()
    _write_json(sp, base)

def _validate_inputs(req: JobCreateRequest) -> None:
    if len(req.genre) > MAX_GENRE_CHARS:
        raise HTTPException(status_code=400, detail=f"genre too long (>{MAX_GENRE_CHARS})")
    if len(req.lyrics) > MAX_LYRICS_CHARS:
        raise HTTPException(status_code=400, detail=f"lyrics too long (>{MAX_LYRICS_CHARS})")

def cleanup_old_jobs() -> None:
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        now = _now()
        for d in JOBS_DIR.iterdir():
            if not d.is_dir():
                continue
            age = now - d.stat().st_mtime
            if age > JOB_TTL_SECONDS:
                shutil.rmtree(d, ignore_errors=True)
                logger.info(f"cleaned old job: {d.name}")
    except Exception as e:
        logger.error(f"cleanup error: {e}")

def _find_latest_recursive(root: Path, pattern: str) -> Optional[Path]:
    if not root.exists():
        return None
    files = list(root.rglob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)

def _iter_audio_files(output_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in OUTPUT_EXTS:
        pattern = f"*{ext}" if ext.startswith(".") else ext
        if not pattern.startswith("*"):
            pattern = "*" + pattern
        files.extend(output_dir.rglob(pattern))
    return files

def _pick_best_mix(output_dir: Path) -> Path:
    """
    '무음' 이슈를 줄이기 위해 mixed 파일을 우선적으로 선택.
    - 1) 파일명에 mixed 포함된 오디오 우선
    - 2) 없으면 stem(vtrack/itrack/vocals/no_vocals 등)로 보이는 파일은 우선 제외
    - 3) 마지막 fallback: 전체 오디오 중 최신
    """
    files = _iter_audio_files(output_dir)
    if not files:
        raise FileNotFoundError(f"No audio generated under: {output_dir}")

    # 1) mixed 우선
    mixed_candidates = [p for p in files if "mixed" in p.name.lower()]
    if mixed_candidates:
        return max(mixed_candidates, key=lambda p: p.stat().st_mtime)

    # 2) stem/중간물로 보이는 것 제외 후 최신
    exclude_tokens = ("vtrack", "itrack", "vocals", "no_vocals", "stem", "stems")
    non_stem = [p for p in files if not any(t in p.name.lower() for t in exclude_tokens)]
    if non_stem:
        return max(non_stem, key=lambda p: p.stat().st_mtime)

    # 3) fallback
    return max(files, key=lambda p: p.stat().st_mtime)

def _media_type_for_path(p: Path) -> str:
    s = p.suffix.lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
    }.get(s, "application/octet-stream")

async def _run(cmd: List[str], *, cwd: Optional[str], timeout: int) -> subprocess.CompletedProcess:
    return await asyncio.to_thread(
        subprocess.run,
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )

def _progress_map(stage: str, pct_0_100: int) -> int:
    pct_0_100 = max(0, min(100, pct_0_100))
    if stage == "yue_stage1":
        return int(pct_0_100 * 0.55)
    if stage == "yue_stage2":
        return int(55 + pct_0_100 * 0.35)
    if stage == "yue_vocoder":
        return min(93, 90 + int(pct_0_100 * 0.03))
    if stage == "demucs":
        return 95
    if stage == "normalize_vocals":
        return 98
    if stage == "done":
        return 100
    return 0

async def _run_streaming_yue(
    cmd: List[str],
    *,
    cwd: str,
    timeout: int,
    job_id: str,
    log_path: Path,
) -> int:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    start = _now()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    current_stage = "yue_stage1"
    last_write_t = 0.0

    _set_status(
        job_id,
        status="running",
        stage=current_stage,
        progress=_progress_map(current_stage, 0),
        started_at=_now(),
    )

    with log_path.open("w", encoding="utf-8") as f:
        while True:
            if proc.stdout is None:
                break

            if _now() - start > timeout:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise TimeoutError(f"YuE timeout > {timeout}s")

            line = await proc.stdout.readline()
            if not line:
                break

            text = line.decode(errors="replace")
            f.write(text)
            f.flush()

            if "Stage1 inference" in text:
                current_stage = "yue_stage1"
            elif "Stage 2 inference" in text or "Stage2 inference" in text:
                current_stage = "yue_stage2"
            elif "Saved:" in text or "Created mix:" in text or "Created mix" in text:
                current_stage = "yue_vocoder"

            pct: Optional[int] = None
            m = _STAGE1_PCT_RE.search(text) or _STAGE2_PCT_RE.search(text)
            if m:
                pct = int(m.group(1))
            else:
                m2 = _FRAC_RE.search(text)
                if m2:
                    a, b = int(m2.group(1)), int(m2.group(2))
                    if b > 0:
                        pct = int(a * 100 / b)

            nowt = _now()
            if nowt - last_write_t >= PROGRESS_WRITE_MIN_INTERVAL:
                fields = {"status": "running", "stage": current_stage}
                if pct is not None:
                    fields["progress"] = _progress_map(current_stage, pct)
                else:
                    fields["progress"] = max(_progress_map(current_stage, 0), 1)
                _set_status(job_id, **fields)
                last_write_t = nowt

    rc = await proc.wait()
    _set_status(job_id, status="running", stage=current_stage, progress=max(_progress_map(current_stage, 100), 1))
    return rc

# ---------------------------
# Core pipeline
# ---------------------------
async def _pipeline(job_id: str) -> None:
    jd = _job_dir(job_id)
    inp = _read_json(_input_path(job_id))

    output_dir = jd / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    genre_file = jd / "genre.txt"
    lyrics_file = jd / "lyrics.txt"
    genre_file.write_text(inp["genre"], encoding="utf-8")
    lyrics_file.write_text(inp["lyrics"], encoding="utf-8")

    infer_cmd = [
        "python", f"{YUE_PATH}/infer.py",
        "--cuda_idx", str(CUDA_IDX),
        "--stage1_model", STAGE1_MODEL,
        "--stage2_model", STAGE2_MODEL,
        "--genre_txt", str(genre_file),
        "--lyrics_txt", str(lyrics_file),
        "--run_n_segments", str(inp["run_n_segments"]),
        "--stage2_batch_size", str(inp["stage2_batch_size"]),
        "--output_dir", str(output_dir),
        "--max_new_tokens", str(inp["max_new_tokens"]),
        "--repetition_penalty", str(inp["repetition_penalty"]),
    ]

    rc = await _run_streaming_yue(
        infer_cmd,
        cwd=YUE_PATH,
        timeout=INFER_TIMEOUT,
        job_id=job_id,
        log_path=jd / "infer_log.txt",
    )
    if rc != 0:
        raise RuntimeError("YuE failed (see infer_log.txt)")

    try:
        mixed = _pick_best_mix(output_dir)
    except Exception as e:
        listing = [str(p) for p in output_dir.rglob("*")][:500]
        _set_status(job_id, status="failed", stage="no_audio", error=str(e), output_listing=listing)
        raise

    _set_status(job_id, stage="yue_done", progress=93, mixed_path=str(mixed))

    warnings: List[str] = []

    # Demucs separation (optional)
    if inp.get("separate_tracks", True):
        _set_status(job_id, stage="demucs", progress=_progress_map("demucs", 0))

        sep_root = output_dir / "separated_demucs"
        sep_root.mkdir(parents=True, exist_ok=True)

        demucs_cmd = [
            "demucs",
            "--two-stems=vocals",
            "-n", DEMUCS_MODEL,
            "-d", DEMUCS_DEVICE,
            "-o", str(sep_root),
            str(mixed),
        ]

        r2 = await _run(demucs_cmd, cwd=str(jd), timeout=DEMUCS_TIMEOUT)
        (jd / "demucs_stdout.txt").write_text(r2.stdout or "", encoding="utf-8")
        (jd / "demucs_stderr.txt").write_text(r2.stderr or "", encoding="utf-8")

        if r2.returncode != 0:
            warnings.append(f"Demucs failed: {(r2.stderr or '').strip()[:300]}")
        else:
            vocals = _find_latest_recursive(sep_root, "vocals.wav")
            mr = _find_latest_recursive(sep_root, "no_vocals.wav")
            if vocals and mr:
                _set_status(job_id, vocals_path=str(vocals), mr_path=str(mr))
            else:
                warnings.append("Demucs finished but vocals/no_vocals not found.")

            # Normalize vocals (optional)
            if vocals and inp.get("normalize_vocals", True):
                _set_status(job_id, stage="normalize_vocals", progress=_progress_map("normalize_vocals", 0))

                norm_dir = output_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)

                sr = int(inp.get("vocal_sr", DEFAULT_VOCAL_SR))
                mono = bool(inp.get("vocal_mono", DEFAULT_VOCAL_MONO))
                norm_vocals = norm_dir / f"vocals_{sr}hz_{'mono' if mono else 'stereo'}.wav"

                ff_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(vocals)]
                if mono:
                    ff_cmd += ["-ac", "1"]
                ff_cmd += ["-ar", str(sr), str(norm_vocals)]

                r3 = await _run(ff_cmd, cwd=str(norm_dir), timeout=FFMPEG_TIMEOUT)
                (jd / "ffmpeg_stdout.txt").write_text(r3.stdout or "", encoding="utf-8")
                (jd / "ffmpeg_stderr.txt").write_text(r3.stderr or "", encoding="utf-8")

                if r3.returncode != 0:
                    warnings.append(f"ffmpeg normalize failed: {(r3.stderr or '').strip()[:300]}")
                else:
                    _set_status(job_id, vocals_norm_path=str(norm_vocals))

    _set_status(
        job_id,
        status="succeeded",
        stage="done",
        progress=100,
        warnings=warnings or None,
        finished_at=_now(),
    )

# ---------------------------
# Worker loop
# ---------------------------
async def worker_loop() -> None:
    logger.info("worker started (single concurrency)")
    while True:
        job_id = await job_queue.get()
        try:
            _set_status(job_id, status="running", stage="picked_by_worker", progress=1)
            await _pipeline(job_id)
            logger.info(f"[{job_id}] succeeded")
        except Exception as e:
            logger.exception(f"[{job_id}] failed")
            _set_status(job_id, status="failed", stage="error", error=str(e), finished_at=_now())
        finally:
            job_queue.task_done()

def _requeue_incomplete_jobs_on_startup() -> int:
    count = 0
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    for d in JOBS_DIR.iterdir():
        if not d.is_dir():
            continue
        sp = d / "status.json"
        ip = d / "input.json"
        if not sp.exists() or not ip.exists():
            continue

        try:
            s = _read_json(sp).get("status")
        except Exception:
            continue

        if s == "queued":
            try:
                job_queue.put_nowait(d.name)
                count += 1
            except asyncio.QueueFull:
                break
        elif s == "running":
            _set_status(d.name, status="failed", stage="restart", error="Server restarted during processing.")
    return count

# ---------------------------
# Routes
# ---------------------------
@app.on_event("startup")
async def on_startup():
    _ensure_dirs()
    cleanup_old_jobs()
    requeued = _requeue_incomplete_jobs_on_startup()
    logger.info(f"requeued jobs: {requeued}")

    global worker_task
    worker_task = asyncio.create_task(worker_loop())

@app.on_event("shutdown")
async def on_shutdown():
    global worker_task
    if worker_task:
        worker_task.cancel()

@app.get("/")
def health():
    return {
        "status": "ok",
        "queue_size": job_queue.qsize(),
        "jobs_dir": str(JOBS_DIR),
        "yue_path": YUE_PATH,
        "stage1_model": STAGE1_MODEL,
        "stage2_model": STAGE2_MODEL,
        "cors_origins": CORS_ORIGINS,
    }

@app.post("/jobs", response_model=JobCreateResponse)
async def create_job(req: JobCreateRequest):
    _validate_inputs(req)

    if job_queue.full():
        raise HTTPException(status_code=429, detail="Queue full, try later.")

    # 충돌 방지
    for _ in range(10):
        job_id = str(uuid.uuid4())[:8]
        jd = _job_dir(job_id)
        if not jd.exists():
            break
    else:
        job_id = str(uuid.uuid4())
        jd = _job_dir(job_id)

    jd.mkdir(parents=True, exist_ok=False)

    _write_json(_input_path(job_id), req.model_dump())
    _set_status(job_id, status="queued", stage="queued", progress=0, created_at=_now())

    await job_queue.put(job_id)

    return JobCreateResponse(job_id=job_id, status="queued", status_url=f"/jobs/{job_id}")

@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    sp = _status_path(job_id)
    if not sp.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return _read_json(sp)

@app.get("/jobs/{job_id}/files/{file_type}")
def download(job_id: str, file_type: str):
    jd = _job_dir(job_id)
    sp = _status_path(job_id)
    if not jd.exists() or not sp.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    st = _read_json(sp)

    if file_type == "output":
        p = st.get("mixed_path")
        if not p:
            raise HTTPException(status_code=404, detail="Output not ready")
        pp = Path(p)
        return FileResponse(str(pp), media_type=_media_type_for_path(pp), filename=f"mixed_{job_id}{pp.suffix}")

    if file_type == "vocal":
        p = st.get("vocals_norm_path") or st.get("vocals_path")
        if not p:
            raise HTTPException(status_code=404, detail="Vocals not ready")
        pp = Path(p)
        return FileResponse(str(pp), media_type=_media_type_for_path(pp), filename=f"vocals_{job_id}{pp.suffix}")

    if file_type == "mr":
        p = st.get("mr_path")
        if not p:
            raise HTTPException(status_code=404, detail="MR not ready")
        pp = Path(p)
        return FileResponse(str(pp), media_type=_media_type_for_path(pp), filename=f"mr_{job_id}{pp.suffix}")

    raise HTTPException(status_code=400, detail="Invalid file_type (output|vocal|mr)")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level=os.environ.get("UVICORN_LOG_LEVEL", "info"),
        access_log=True,
    )
