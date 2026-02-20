"""
Suno Provider - External API-based music generation using Suno API.
"""

import os
import asyncio
import logging
import aiohttp
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Literal

from providers.audio_utils import post_process_audio

logger = logging.getLogger("yue-api.suno-provider")

# ---------------------------
# Suno API Config
# ---------------------------
SUNO_API_BASE = os.environ.get("SUNO_API_BASE", "https://api.sunoapi.org/api/v1")
SUNO_API_TOKEN = os.environ.get("SUNO_API_TOKEN", "")
SUNO_POLL_INTERVAL = int(os.environ.get("SUNO_POLL_INTERVAL", "10"))  # seconds
SUNO_TIMEOUT = int(os.environ.get("SUNO_TIMEOUT", "600"))  # 10 minutes max

# Callback URL base (must be publicly accessible for Suno to reach)
CALLBACK_BASE_URL = os.environ.get("CALLBACK_BASE_URL", "")


def _now() -> float:
    import time
    return time.time()


def _get_headers() -> Dict[str, str]:
    """Get authorization headers for Suno API."""
    if not SUNO_API_TOKEN:
        raise ValueError("SUNO_API_TOKEN environment variable is not set")
    return {
        "Authorization": f"Bearer {SUNO_API_TOKEN}",
        "Content-Type": "application/json",
    }


async def generate_music(
    job_id: str,
    job_dir: Path,
    inp: Dict[str, Any],
    set_status: Callable[..., None],
) -> Dict[str, Any]:
    """
    Suno API를 통해 음악 생성.
    """
    set_status(
        job_id,
        status="running",
        stage="suno_generate",
        progress=5,
        started_at=_now(),
    )
    
    # Build Suno API request payload
    payload = {
        "customMode": inp.get("custom_mode", True),
        "instrumental": inp.get("instrumental", False),
        "model": inp.get("model", "V5"),
        "prompt": inp.get("prompt", ""),
        "style": inp.get("style", ""),
        "title": inp.get("title", ""),
    }
    
    # Optional fields
    # Auto-generate callback URL if CALLBACK_BASE_URL is set
    callback_url = inp.get("callback_url")
    if not callback_url and CALLBACK_BASE_URL:
        callback_url = f"{CALLBACK_BASE_URL.rstrip('/')}/jobs/callback"
    if callback_url:
        payload["callBackUrl"] = callback_url
    if inp.get("persona_id"):
        payload["personaId"] = inp["persona_id"]
    if inp.get("negative_tags"):
        payload["negativeTags"] = inp["negative_tags"]
    if inp.get("vocal_gender"):
        payload["vocalGender"] = inp["vocal_gender"]
    if inp.get("style_weight") is not None:
        payload["styleWeight"] = inp["style_weight"]
    if inp.get("weirdness_constraint") is not None:
        payload["weirdnessConstraint"] = inp["weirdness_constraint"]
    if inp.get("audio_weight") is not None:
        payload["audioWeight"] = inp["audio_weight"]
    
    url = f"{SUNO_API_BASE}/generate"
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=_get_headers()) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Suno generate failed: {resp.status} - {error_text}")
            
            result = await resp.json()
    
    # Check for API-level error in response body
    if isinstance(result, dict) and result.get("code") and result.get("code") != 200:
        error_msg = result.get("msg", "Unknown Suno API error")
        raise RuntimeError(f"Suno API error: {result.get('code')} - {error_msg}")
            
    set_status(
        job_id,
        status="running",
        stage="suno_submitted",
        progress=10,
        suno_response=result,
    )
    
    return result


async def extend_music(
    job_id: str,
    job_dir: Path,
    inp: Dict[str, Any],
    set_status: Callable[..., None],
) -> Dict[str, Any]:
    """
    Suno API를 통해 음악 연장.
    """
    if not inp.get("audio_id"):
        raise ValueError("audio_id is required for extend")
    
    set_status(
        job_id,
        status="running",
        stage="suno_extend",
        progress=5,
        started_at=_now(),
    )
    
    payload = {
        "defaultParamFlag": inp.get("default_param_flag", True),
        "audioId": inp["audio_id"],
        "model": inp.get("model", "V5"),
    }
    
    # Optional fields
    if inp.get("callback_url"):
        payload["callBackUrl"] = inp["callback_url"]
    if inp.get("prompt"):
        payload["prompt"] = inp["prompt"]
    if inp.get("style"):
        payload["style"] = inp["style"]
    if inp.get("title"):
        payload["title"] = inp["title"]
    if inp.get("continue_at") is not None:
        payload["continueAt"] = inp["continue_at"]
    if inp.get("persona_id"):
        payload["personaId"] = inp["persona_id"]
    if inp.get("negative_tags"):
        payload["negativeTags"] = inp["negative_tags"]
    if inp.get("vocal_gender"):
        payload["vocalGender"] = inp["vocal_gender"]
    if inp.get("style_weight") is not None:
        payload["styleWeight"] = inp["style_weight"]
    if inp.get("weirdness_constraint") is not None:
        payload["weirdnessConstraint"] = inp["weirdness_constraint"]
    if inp.get("audio_weight") is not None:
        payload["audioWeight"] = inp["audio_weight"]
    
    url = f"{SUNO_API_BASE}/generate/extend"
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=_get_headers()) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Suno extend failed: {resp.status} - {error_text}")
            
            result = await resp.json()
    
    # Check for API-level error in response body
    if isinstance(result, dict) and result.get("code") and result.get("code") != 200:
        error_msg = result.get("msg", "Unknown Suno API error")
        raise RuntimeError(f"Suno extend API error: {result.get('code')} - {error_msg}")
    
    set_status(
        job_id,
        status="running",
        stage="suno_extend_submitted",
        progress=10,
        suno_response=result,
    )
    
    return result


async def get_record_info() -> Dict[str, Any]:
    """
    Suno API에서 음악 정보 조회.
    """
    url = f"{SUNO_API_BASE}/generate/record-info"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=_get_headers()) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Suno record-info failed: {resp.status} - {error_text}")
            
            return await resp.json()


async def poll_and_download(
    job_id: str,
    job_dir: Path,
    suno_task_id: str,
    inp: Dict[str, Any],
    set_status: Callable[..., None],
) -> Path:
    """
    Suno 작업이 완료될 때까지 polling하고 결과 다운로드.
    """
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = _now()
    poll_count = 0
    
    async with aiohttp.ClientSession() as session:
        while True:
            if _now() - start_time > SUNO_TIMEOUT:
                raise TimeoutError(f"Suno polling timeout > {SUNO_TIMEOUT}s")
            
            poll_count += 1
            progress = min(10 + poll_count * 5, 80)
            
            # Get record info to check status
            url = f"{SUNO_API_BASE}/generate/record-info"
            async with session.get(url, headers=_get_headers()) as resp:
                if resp.status != 200:
                    await asyncio.sleep(SUNO_POLL_INTERVAL)
                    continue
                
                records = await resp.json()
            
            # Find our task in records
            task_info = None
            if isinstance(records, dict) and "data" in records:
                for record in records.get("data", []):
                    if record.get("id") == suno_task_id or record.get("taskId") == suno_task_id:
                        task_info = record
                        break
            
            if task_info:
                status = task_info.get("status", "").lower()
                
                if status in ("completed", "success", "done"):
                    set_status(job_id, stage="suno_downloading", progress=85)
                    
                    # Download audio
                    audio_url = task_info.get("audioUrl") or task_info.get("audio_url")
                    if audio_url:
                        audio_path = output_dir / f"suno_{suno_task_id}.mp3"
                        async with session.get(audio_url) as audio_resp:
                            if audio_resp.status == 200:
                                audio_path.write_bytes(await audio_resp.read())
                                set_status(
                                    job_id,
                                    stage="suno_downloaded",
                                    progress=90,
                                    mixed_path=str(audio_path),
                                    suno_info=task_info,
                                )
                                return audio_path
                    
                    raise RuntimeError("Suno completed but no audio URL found")
                
                elif status in ("failed", "error"):
                    error_msg = task_info.get("error") or task_info.get("message") or "Unknown error"
                    raise RuntimeError(f"Suno generation failed: {error_msg}")
                
                else:
                    set_status(
                        job_id,
                        stage=f"suno_processing",
                        progress=progress,
                        suno_status=status,
                    )
            
            await asyncio.sleep(SUNO_POLL_INTERVAL)


async def run_suno_pipeline(
    job_id: str,
    job_dir: Path,
    inp: Dict[str, Any],
    set_status: Callable[..., None],
    action: Literal["generate", "extend"] = "generate",
) -> None:
    """
    Suno API 파이프라인 실행.
    
    Args:
        job_id: 작업 ID
        job_dir: 작업 디렉토리
        inp: 입력 파라미터
        set_status: 상태 업데이트 콜백
        action: 수행할 작업 (generate 또는 extend)
    """
    output_dir = job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Call appropriate Suno API
    if action == "extend":
        result = await extend_music(job_id, job_dir, inp, set_status)
    else:
        result = await generate_music(job_id, job_dir, inp, set_status)
    
    # Extract task ID from response and save for callback routing
    suno_task_id = None
    if isinstance(result, dict):
        suno_task_id = result.get("id") or result.get("taskId") or result.get("data", {}).get("taskId") or result.get("data", {}).get("id")
    
    # Check if callback URL was used (either provided or auto-generated)
    callback_url = inp.get("callback_url")
    if not callback_url and CALLBACK_BASE_URL:
        callback_url = f"{CALLBACK_BASE_URL.rstrip('/')}/jobs/callback"
    
    # If callback URL is set, mark as waiting for callback
    if callback_url:
        set_status(
            job_id,
            status="waiting_callback",
            stage="suno_callback_pending",
            progress=15,
            suno_response=result,
            suno_task_id=suno_task_id,
            message="Waiting for Suno callback.",
        )
        return
    
    if not suno_task_id:
        logger.warning(f"Could not extract Suno task ID from response: {result}")
        set_status(
            job_id,
            status="succeeded",
            stage="suno_submitted",
            progress=100,
            suno_response=result,
            message="Suno request submitted. Check record-info for status.",
            finished_at=_now(),
        )
        return
    
    # Poll and download result
    audio_path = await poll_and_download(job_id, job_dir, suno_task_id, inp, set_status)
    
    # 공통 후처리 (Demucs 분리 + 보컬 정규화)
    warnings = await post_process_audio(
        audio_path=audio_path,
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
