"""
Shared utilities for audio post-processing.
Used by both YuE and Suno providers.
"""

import os
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple

logger = logging.getLogger("yue-api.audio-utils")

# ---------------------------
# Config
# ---------------------------
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs")
DEMUCS_DEVICE = os.environ.get("DEMUCS_DEVICE", "cuda")
DEMUCS_TIMEOUT = int(os.environ.get("DEMUCS_TIMEOUT", "900"))
FFMPEG_TIMEOUT = int(os.environ.get("FFMPEG_TIMEOUT", "300"))
DEFAULT_VOCAL_SR = int(os.environ.get("DEFAULT_VOCAL_SR", "16000"))
DEFAULT_VOCAL_MONO = os.environ.get("DEFAULT_VOCAL_MONO", "1") == "1"


def _now() -> float:
    import time
    return time.time()


def _find_latest_recursive(root: Path, pattern: str) -> Optional[Path]:
    if not root.exists():
        return None
    files = list(root.rglob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


async def _run(cmd: List[str], *, cwd: Optional[str], timeout: int) -> subprocess.CompletedProcess:
    return await asyncio.to_thread(
        subprocess.run,
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


async def run_demucs_separation(
    audio_path: Path,
    output_dir: Path,
    job_dir: Path,
    job_id: str,
    set_status: Callable[..., None],
) -> Tuple[Optional[Path], Optional[Path], List[str]]:
    """
    Demucs를 사용하여 보컬/MR 분리.
    
    Args:
        audio_path: 분리할 오디오 파일 경로
        output_dir: 결과 저장 디렉토리
        job_dir: 작업 디렉토리 (로그 저장용)
        job_id: 작업 ID
        set_status: 상태 업데이트 콜백
    
    Returns:
        (vocals_path, mr_path, warnings)
    """
    warnings: List[str] = []
    vocals: Optional[Path] = None
    mr: Optional[Path] = None
    
    set_status(job_id, stage="demucs", progress=95)
    
    sep_root = output_dir / "separated_demucs"
    sep_root.mkdir(parents=True, exist_ok=True)
    
    demucs_cmd = [
        "demucs",
        "--two-stems=vocals",
        "-n", DEMUCS_MODEL,
        "-d", DEMUCS_DEVICE,
        "-o", str(sep_root),
        str(audio_path),
    ]
    
    r = await _run(demucs_cmd, cwd=str(job_dir), timeout=DEMUCS_TIMEOUT)
    (job_dir / "demucs_stdout.txt").write_text(r.stdout or "", encoding="utf-8")
    (job_dir / "demucs_stderr.txt").write_text(r.stderr or "", encoding="utf-8")
    
    if r.returncode != 0:
        warnings.append(f"Demucs failed: {(r.stderr or '').strip()[:300]}")
    else:
        vocals = _find_latest_recursive(sep_root, "vocals.wav")
        mr = _find_latest_recursive(sep_root, "no_vocals.wav")
        if vocals and mr:
            set_status(job_id, vocals_path=str(vocals), mr_path=str(mr))
        else:
            warnings.append("Demucs finished but vocals/no_vocals not found.")
    
    return vocals, mr, warnings


async def normalize_vocals(
    vocals_path: Path,
    output_dir: Path,
    job_dir: Path,
    job_id: str,
    vocal_sr: int = DEFAULT_VOCAL_SR,
    vocal_mono: bool = DEFAULT_VOCAL_MONO,
    set_status: Optional[Callable[..., None]] = None,
) -> Tuple[Optional[Path], List[str]]:
    """
    보컬 트랙을 정규화 (샘플레이트, 모노 변환).
    
    Args:
        vocals_path: 원본 보컬 파일 경로
        output_dir: 결과 저장 디렉토리
        job_dir: 작업 디렉토리
        job_id: 작업 ID
        vocal_sr: 타겟 샘플레이트
        vocal_mono: 모노 변환 여부
        set_status: 상태 업데이트 콜백
    
    Returns:
        (normalized_path, warnings)
    """
    warnings: List[str] = []
    
    if set_status:
        set_status(job_id, stage="normalize_vocals", progress=98)
    
    norm_dir = output_dir / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)
    
    norm_vocals = norm_dir / f"vocals_{vocal_sr}hz_{'mono' if vocal_mono else 'stereo'}.wav"
    
    ff_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(vocals_path)]
    if vocal_mono:
        ff_cmd += ["-ac", "1"]
    ff_cmd += ["-ar", str(vocal_sr), str(norm_vocals)]
    
    r = await _run(ff_cmd, cwd=str(norm_dir), timeout=FFMPEG_TIMEOUT)
    (job_dir / "ffmpeg_stdout.txt").write_text(r.stdout or "", encoding="utf-8")
    (job_dir / "ffmpeg_stderr.txt").write_text(r.stderr or "", encoding="utf-8")
    
    if r.returncode != 0:
        warnings.append(f"ffmpeg normalize failed: {(r.stderr or '').strip()[:300]}")
        return None, warnings
    
    return norm_vocals, warnings


async def post_process_audio(
    audio_path: Path,
    output_dir: Path,
    job_dir: Path,
    job_id: str,
    inp: Dict[str, Any],
    set_status: Callable[..., None],
) -> List[str]:
    """
    오디오 후처리 (Demucs 분리 + 보컬 정규화).
    YuE와 Suno 모두에서 공통으로 사용.
    
    Args:
        audio_path: 처리할 오디오 파일 경로
        output_dir: 결과 저장 디렉토리
        job_dir: 작업 디렉토리
        job_id: 작업 ID
        inp: 입력 파라미터 (separate_tracks, normalize_vocals 등)
        set_status: 상태 업데이트 콜백
    
    Returns:
        경고 메시지 목록
    """
    warnings: List[str] = []
    
    if not inp.get("separate_tracks", True):
        return warnings
    
    vocals, mr, demucs_warnings = await run_demucs_separation(
        audio_path, output_dir, job_dir, job_id, set_status
    )
    warnings.extend(demucs_warnings)
    
    if vocals and inp.get("normalize_vocals", True):
        vocal_sr = int(inp.get("vocal_sr", DEFAULT_VOCAL_SR))
        vocal_mono = bool(inp.get("vocal_mono", DEFAULT_VOCAL_MONO))
        
        norm_vocals, norm_warnings = await normalize_vocals(
            vocals, output_dir, job_dir, job_id,
            vocal_sr=vocal_sr, vocal_mono=vocal_mono, set_status=set_status
        )
        warnings.extend(norm_warnings)
        
        if norm_vocals:
            set_status(job_id, vocals_norm_path=str(norm_vocals))
    
    return warnings
