"""
YuE Provider - Local GPU-based music generation using YuE model.
Extracted from original api_server.py pipeline logic.
"""

import os
import re
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

from providers.audio_utils import post_process_audio

logger = logging.getLogger("yue-api.yue-provider")

# ---------------------------
# Config (env override)
# ---------------------------
YUE_PATH = os.environ.get("YUE_PATH", "/opt/YuE/inference")
CUDA_IDX = os.environ.get("CUDA_IDX", "0")
STAGE1_MODEL = os.environ.get("STAGE1_MODEL", "m-a-p/YuE-s1-7B-anneal-jp-kr-icl")
STAGE2_MODEL = os.environ.get("STAGE2_MODEL", "m-a-p/YuE-s2-1B-general")

INFER_TIMEOUT = int(os.environ.get("INFER_TIMEOUT", "1800"))
PROGRESS_WRITE_MIN_INTERVAL = float(os.environ.get("PROGRESS_WRITE_MIN_INTERVAL", "1.0"))

OUTPUT_EXTS = tuple(x.strip() for x in os.environ.get(
    "OUTPUT_EXTS",
    ".wav,.mp3,.flac,.m4a,.ogg"
).split(",") if x.strip())

# ---------------------------
# Stage trace (stdout parsing)
# ---------------------------
_STAGE1_PCT_RE = re.compile(r"Stage1 inference\.\.\.:\s*(\d+)%")
_STAGE2_PCT_RE = re.compile(r"Stage\s*2 inference\.\.\.\s*(\d+)%")
_FRAC_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _now() -> float:
    import time
    return time.time()


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
    """
    files = _iter_audio_files(output_dir)
    if not files:
        raise FileNotFoundError(f"No audio generated under: {output_dir}")

    mixed_candidates = [p for p in files if "mixed" in p.name.lower()]
    if mixed_candidates:
        return max(mixed_candidates, key=lambda p: p.stat().st_mtime)

    exclude_tokens = ("vtrack", "itrack", "vocals", "no_vocals", "stem", "stems")
    non_stem = [p for p in files if not any(t in p.name.lower() for t in exclude_tokens)]
    if non_stem:
        return max(non_stem, key=lambda p: p.stat().st_mtime)

    return max(files, key=lambda p: p.stat().st_mtime)


async def _run_streaming_yue(
    cmd: List[str],
    *,
    cwd: str,
    timeout: int,
    job_id: str,
    log_path: Path,
    set_status: Callable[..., None],
) -> int:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        limit=10 * 1024 * 1024,
    )

    start = _now()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    current_stage = "yue_stage1"
    last_write_t = 0.0

    set_status(
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
                set_status(job_id, **fields)
                last_write_t = nowt

    rc = await proc.wait()
    set_status(job_id, status="running", stage=current_stage, progress=max(_progress_map(current_stage, 100), 1))
    return rc


async def run_yue_pipeline(
    job_id: str,
    job_dir: Path,
    inp: Dict[str, Any],
    set_status: Callable[..., None],
) -> None:
    """
    YuE 로컬 생성 파이프라인 실행.
    
    Args:
        job_id: 작업 ID
        job_dir: 작업 디렉토리
        inp: 입력 파라미터 (genre, lyrics, ...)
        set_status: 상태 업데이트 콜백 함수
    """
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    genre_file = job_dir / "genre.txt"
    lyrics_file = job_dir / "lyrics.txt"
    genre_file.write_text(inp["genre"], encoding="utf-8")
    lyrics_file.write_text(inp["lyrics"], encoding="utf-8")

    infer_cmd = [
        "python", f"{YUE_PATH}/infer.py",
        "--cuda_idx", str(CUDA_IDX),
        "--stage1_model", STAGE1_MODEL,
        "--stage2_model", STAGE2_MODEL,
        "--genre_txt", str(genre_file),
        "--lyrics_txt", str(lyrics_file),
        "--run_n_segments", str(inp.get("run_n_segments", 2)),
        "--stage2_batch_size", str(inp.get("stage2_batch_size", 4)),
        "--output_dir", str(output_dir),
        "--max_new_tokens", str(inp.get("max_new_tokens", 3000)),
        "--repetition_penalty", str(inp.get("repetition_penalty", 1.1)),
    ]

    rc = await _run_streaming_yue(
        infer_cmd,
        cwd=YUE_PATH,
        timeout=INFER_TIMEOUT,
        job_id=job_id,
        log_path=job_dir / "infer_log.txt",
        set_status=set_status,
    )
    if rc != 0:
        raise RuntimeError("YuE failed (see infer_log.txt)")

    try:
        mixed = _pick_best_mix(output_dir)
    except Exception as e:
        listing = [str(p) for p in output_dir.rglob("*")][:500]
        set_status(job_id, status="failed", stage="no_audio", error=str(e), output_listing=listing)
        raise

    set_status(job_id, stage="yue_done", progress=93, mixed_path=str(mixed))

    # 공통 후처리 (Demucs 분리 + 보컬 정규화)
    warnings = await post_process_audio(
        audio_path=mixed,
        output_dir=output_dir,
        job_dir=job_dir,
        job_id=job_id,
        inp=inp,
        set_status=set_status,
    )

    set_status(
        job_id,
        status="succeeded",
        stage="done",
        progress=100,
        warnings=warnings or None,
        finished_at=_now(),
    )
