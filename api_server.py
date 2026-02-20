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
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal

from providers import ProviderType
from providers.yue_provider import run_yue_pipeline
from providers.suno_provider import run_suno_pipeline, get_record_info

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
JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/tmp/yue_jobs"))
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", "3600"))

MAX_GENRE_CHARS = int(os.environ.get("MAX_GENRE_CHARS", "2000"))
MAX_LYRICS_CHARS = int(os.environ.get("MAX_LYRICS_CHARS", "20000"))

QUEUE_MAXSIZE = int(os.environ.get("QUEUE_MAXSIZE", "100"))

# Callback URL base (for Suno callbacks)
CALLBACK_BASE_URL = os.environ.get("CALLBACK_BASE_URL", "")

# Suno rate limiting (requests per minute)
SUNO_RATE_LIMIT = int(os.environ.get("SUNO_RATE_LIMIT", "10"))

# CORS origins (comma separated)
CORS_ORIGINS = ["*"]

# ---------------------------
# Rate Limiter (Sliding Window)
# ---------------------------
class RateLimiter:
    """분당 요청 수를 제한하는 Rate Limiter (Sliding Window)"""
    
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.timestamps: List[float] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """요청 허용까지 대기 (필요시)"""
        async with self._lock:
            while True:
                now = time.time()
                # 윈도우 밖의 오래된 타임스탬프 제거
                self.timestamps = [t for t in self.timestamps if now - t < self.window]
                
                if len(self.timestamps) < self.max_requests:
                    self.timestamps.append(now)
                    return
                
                # 가장 오래된 요청이 만료될 때까지 대기
                wait_time = self.window - (now - self.timestamps[0]) + 0.1
                logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

# ---------------------------
# App + in-process queues
# ---------------------------
app = FastAPI(
    title="Music Generation API",
    version="3.0.0",
    description="Multi-provider music generation: YuE (local GPU) and Suno (external API)",
)

# ---------------------------
# CORS (after app creation)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dual queue system
yue_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
suno_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
suno_rate_limiter = RateLimiter(max_requests=SUNO_RATE_LIMIT, window_seconds=60)

yue_worker_task: Optional[asyncio.Task] = None
suno_worker_task: Optional[asyncio.Task] = None

# ---------------------------
# Request/Response models
# ---------------------------
class JobCreateRequest(BaseModel):
    """음악 생성 요청 (YuE 또는 Suno)"""
    provider: Literal["yue", "suno"] = Field("yue", description="Provider: 'yue' (local) or 'suno' (API)")
    
    # === YuE 전용 필드 ===
    genre: Optional[str] = Field(None, description="[YuE] Genre description")
    lyrics: Optional[str] = Field(None, description="[YuE] Song lyrics")
    run_n_segments: int = Field(2, ge=1, le=12, description="[YuE] Number of segments")
    max_new_tokens: int = Field(3000, ge=128, le=10000, description="[YuE] Max tokens")
    repetition_penalty: float = Field(1.1, ge=0.8, le=2.0, description="[YuE] Repetition penalty")
    stage2_batch_size: int = Field(4, ge=1, le=32, description="[YuE] Stage2 batch size")
    
    # === 공통 필드 (Demucs 후처리) ===
    separate_tracks: bool = Field(True, description="Run Demucs vocal/MR separation")
    normalize_vocals: bool = Field(True, description="Normalize vocals for lipsync")
    vocal_sr: int = Field(16000, ge=8000, le=48000, description="Vocal sample rate")
    vocal_mono: bool = Field(True, description="Mono vocals")
    
    # === Suno 전용 필드 ===
    prompt: Optional[str] = Field(None, description="[Suno] Music generation prompt")
    style: Optional[str] = Field(None, description="[Suno] Music style (e.g., 'Classical')")
    title: Optional[str] = Field(None, description="[Suno] Song title")
    instrumental: bool = Field(False, description="[Suno] Instrumental only (no vocals)")
    model: str = Field("V4_5ALL", description="[Suno] Model version")
    callback_url: Optional[str] = Field(None, description="[Suno] Callback URL for async result")
    persona_id: Optional[str] = Field(None, description="[Suno] Persona ID")
    negative_tags: Optional[str] = Field(None, description="[Suno] Negative tags")
    vocal_gender: Optional[str] = Field("f", description="[Suno] Vocal gender ('m' or 'f')")
    custom_mode: bool = Field(True, description="[Suno] Custom mode")
    style_weight: Optional[float] = Field(None, ge=0, le=1, description="[Suno] Style weight")
    weirdness_constraint: Optional[float] = Field(None, ge=0, le=1, description="[Suno] Weirdness constraint")
    audio_weight: Optional[float] = Field(None, ge=0, le=1, description="[Suno] Audio weight")


class ExtendRequest(BaseModel):
    """Suno 음악 연장 요청"""
    audio_id: str = Field(..., description="Suno audio ID to extend")
    model: str = Field("V4_5ALL", description="Model version")
    callback_url: Optional[str] = Field(None, description="Callback URL")
    prompt: Optional[str] = Field(None, description="Extension prompt")
    style: Optional[str] = Field(None, description="Music style")
    title: Optional[str] = Field(None, description="Extended song title")
    continue_at: Optional[int] = Field(None, description="Continue at position (seconds)")
    persona_id: Optional[str] = Field(None, description="Persona ID")
    negative_tags: Optional[str] = Field(None, description="Negative tags")
    vocal_gender: Optional[str] = Field("f", description="Vocal gender ('m' or 'f')")
    default_param_flag: bool = Field(True, description="Use default parameters")
    style_weight: Optional[float] = Field(None, ge=0, le=1)
    weirdness_constraint: Optional[float] = Field(None, ge=0, le=1)
    audio_weight: Optional[float] = Field(None, ge=0, le=1)
    
    # === 공통 필드 (Demucs 후처리) ===
    separate_tracks: bool = Field(True, description="Run Demucs vocal/MR separation")
    normalize_vocals: bool = Field(True, description="Normalize vocals for lipsync")
    vocal_sr: int = Field(16000, ge=8000, le=48000, description="Vocal sample rate")
    vocal_mono: bool = Field(True, description="Mono vocals")


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    status_url: str
    provider: str


class SunoCallbackAudioItem(BaseModel):
    """Suno 콜백에서 수신하는 개별 오디오 정보"""
    id: str
    audio_url: str
    source_audio_url: Optional[str] = None
    image_url: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[float] = None
    prompt: Optional[str] = None
    tags: Optional[str] = None
    model_name: Optional[str] = None


class SunoCallbackData(BaseModel):
    """Suno 콜백 data 필드"""
    callbackType: str
    task_id: str
    data: List[SunoCallbackAudioItem]


class SunoCallbackRequest(BaseModel):
    """Suno API 콜백 요청"""
    code: int
    msg: str
    data: Optional[SunoCallbackData] = None

# ---------------------------
# Helpers
# ---------------------------
def _now() -> float:
    return time.time()

def _ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    infer_py = Path(YUE_PATH) / "infer.py"
    if not infer_py.exists():
        logger.warning(f"YuE infer.py not found at {infer_py} - YuE provider may not work")

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

    if "progress" in fields:
        prev = int(base.get("progress", 0) or 0)
        fields["progress"] = max(prev, int(fields["progress"] or 0))

    base.update(fields)
    base["updated_at"] = _now()
    _write_json(sp, base)

def _validate_yue_inputs(req: JobCreateRequest) -> None:
    if not req.genre:
        raise HTTPException(status_code=400, detail="genre is required for YuE provider")
    if not req.lyrics:
        raise HTTPException(status_code=400, detail="lyrics is required for YuE provider")
    if len(req.genre) > MAX_GENRE_CHARS:
        raise HTTPException(status_code=400, detail=f"genre too long (>{MAX_GENRE_CHARS})")
    if len(req.lyrics) > MAX_LYRICS_CHARS:
        raise HTTPException(status_code=400, detail=f"lyrics too long (>{MAX_LYRICS_CHARS})")

def _validate_suno_inputs(req: JobCreateRequest) -> None:
    if not req.prompt and not req.style:
        raise HTTPException(status_code=400, detail="prompt or style is required for Suno provider")

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

def _media_type_for_path(p: Path) -> str:
    s = p.suffix.lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
    }.get(s, "application/octet-stream")

# ---------------------------
# Core pipeline (dispatcher)
# ---------------------------
async def _pipeline(job_id: str) -> None:
    jd = _job_dir(job_id)
    inp = _read_json(_input_path(job_id))
    provider = inp.get("provider", "yue")
    
    if provider == "yue":
        await run_yue_pipeline(job_id, jd, inp, _set_status)
    elif provider == "suno":
        action = inp.get("action", "generate")
        await run_suno_pipeline(job_id, jd, inp, _set_status, action=action)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ---------------------------
# Worker loops (YuE sequential, Suno concurrent)
# ---------------------------

async def yue_worker_loop() -> None:
    """YuE 작업 순차 처리 (GPU 사용)"""
    logger.info("YuE worker started (sequential)")
    while True:
        job_id = await yue_queue.get()
        try:
            _set_status(job_id, status="running", stage="picked_by_yue_worker", progress=1)
            await _pipeline(job_id)
            logger.info(f"[{job_id}] YuE job ongoing")
        except Exception as e:
            logger.exception(f"[{job_id}] YuE job failed")
            _set_status(job_id, status="failed", stage="error", error=str(e), finished_at=_now())
        finally:
            yue_queue.task_done()


async def _process_suno_job(job_id: str) -> None:
    """단일 Suno 작업 처리"""
    try:
        _set_status(job_id, status="running", stage="picked_by_suno_worker", progress=1)
        await _pipeline(job_id)
        logger.info(f"[{job_id}] Suno job ongoing")
    except Exception as e:
        logger.exception(f"[{job_id}] Suno job failed")
        _set_status(job_id, status="failed", stage="error", error=str(e), finished_at=_now())
    finally:
        suno_queue.task_done()


async def suno_worker_loop() -> None:
    """Suno 작업 동시 처리 (Rate Limit: 분당 10회)"""
    logger.info(f"Suno worker started (rate limit: {SUNO_RATE_LIMIT}/min)")
    while True:
        job_id = await suno_queue.get()
        # Rate limit 적용 후 비동기 태스크로 실행
        await suno_rate_limiter.acquire()
        asyncio.create_task(_process_suno_job(job_id))


def _requeue_incomplete_jobs_on_startup() -> int:
    """서버 재시작 시 미완료 작업 다시 큐에 추가"""
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
            st = _read_json(sp)
            status = st.get("status")
            provider = st.get("provider", "yue")
        except Exception:
            continue

        if status == "queued":
            try:
                if provider == "suno":
                    suno_queue.put_nowait(d.name)
                else:
                    yue_queue.put_nowait(d.name)
                count += 1
            except asyncio.QueueFull:
                break
        elif status == "running":
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

    global yue_worker_task, suno_worker_task
    yue_worker_task = asyncio.create_task(yue_worker_loop())
    suno_worker_task = asyncio.create_task(suno_worker_loop())

@app.on_event("shutdown")
async def on_shutdown():
    global yue_worker_task, suno_worker_task
    if yue_worker_task:
        yue_worker_task.cancel()
    if suno_worker_task:
        suno_worker_task.cancel()

@app.get("/")
def health():
    return {
        "status": "ok",
        "version": "3.1.0",
        "providers": ["yue", "suno"],
        "yue_queue_size": yue_queue.qsize(),
        "suno_queue_size": suno_queue.qsize(),
        "suno_rate_limit": SUNO_RATE_LIMIT,
        "jobs_dir": str(JOBS_DIR),
        "yue_path": YUE_PATH,
    }

@app.post("/jobs", response_model=JobCreateResponse)
async def create_job(req: JobCreateRequest):
    """
    음악 생성 작업 생성.
    
    - provider=yue: 로컬 GPU에서 YuE 모델로 생성 (genre, lyrics 필수)
    - provider=suno: Suno API로 생성 (prompt 또는 style 필수)
    """
    # Validate based on provider
    if req.provider == "yue":
        _validate_yue_inputs(req)
    elif req.provider == "suno":
        _validate_suno_inputs(req)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}")

    # Provider별 큐 선택
    target_queue = suno_queue if req.provider == "suno" else yue_queue
    
    if target_queue.full():
        raise HTTPException(status_code=429, detail="Queue full, try later.")

    # Generate unique job ID
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
    _set_status(job_id, status="queued", stage="queued", progress=0, created_at=_now(), provider=req.provider)

    await target_queue.put(job_id)

    return JobCreateResponse(
        job_id=job_id,
        status="queued",
        status_url=f"/jobs/{job_id}",
        provider=req.provider,
    )

@app.post("/jobs/extend", response_model=JobCreateResponse, include_in_schema=False)
async def extend_job(req: ExtendRequest):
    """
    Suno 음악 연장 작업 생성.
    
    기존 Suno 음악의 audio_id를 사용하여 음악을 연장합니다.
    """
    if suno_queue.full():
        raise HTTPException(status_code=429, detail="Queue full, try later.")

    for _ in range(10):
        job_id = str(uuid.uuid4())[:8]
        jd = _job_dir(job_id)
        if not jd.exists():
            break
    else:
        job_id = str(uuid.uuid4())
        jd = _job_dir(job_id)

    jd.mkdir(parents=True, exist_ok=False)

    input_data = req.model_dump()
    input_data["provider"] = "suno"
    input_data["action"] = "extend"
    
    _write_json(_input_path(job_id), input_data)
    _set_status(job_id, status="queued", stage="queued", progress=0, created_at=_now(), provider="suno")

    await suno_queue.put(job_id)

    return JobCreateResponse(
        job_id=job_id,
        status="queued",
        status_url=f"/jobs/{job_id}",
        provider="suno",
    )

@app.get("/jobs/suno/record-info")
async def suno_record_info():
    """
    Suno API에서 음악 레코드 정보 조회.
    
    현재 계정의 생성된 음악 목록을 반환합니다.
    """
    try:
        return await get_record_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jobs/callback")
async def suno_callback(req: SunoCallbackRequest):
    """
    Suno 콜백 수신 엔드포인트.
    
    Suno API가 음악 생성 완료 시 이 엔드포인트로 결과를 전송합니다.
    task_id로 job을 찾아 오디오 다운로드 및 후처리를 진행합니다.
    """
    import aiohttp
    from providers.audio_utils import post_process_audio
    
    if req.code != 200 or not req.data:
        logger.warning(f"Suno callback with error: {req.code} - {req.msg}")
        return {"status": "ignored", "reason": f"code={req.code}, msg={req.msg}"}
    
    suno_task_id = req.data.task_id
    callback_type = req.data.callbackType
    
    # Find job by suno_task_id
    job_id = None
    for d in JOBS_DIR.iterdir():
        if not d.is_dir():
            continue
        sp = d / "status.json"
        if not sp.exists():
            continue
        try:
            st = _read_json(sp)
            if st.get("suno_task_id") == suno_task_id:
                job_id = d.name
                break
        except Exception:
            continue
    
    if not job_id:
        logger.warning(f"Suno callback: no job found for task_id={suno_task_id}")
        return {"status": "ignored", "reason": "job not found"}
    
    jd = _job_dir(job_id)
    output_dir = jd / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if callback_type != "complete":
        _set_status(job_id, stage=f"suno_{callback_type}", suno_callback=req.model_dump())
        return {"status": "received", "job_id": job_id, "callback_type": callback_type}
    
    # Download and process ALL audio results
    audio_items = req.data.data
    if not audio_items:
        _set_status(job_id, status="failed", stage="callback_no_audio", error="No audio in callback")
        return {"status": "error", "reason": "no audio data"}
    
    _set_status(
        job_id,
        stage="downloading",
        progress=80,
        suno_all_results=[item.model_dump() for item in audio_items],
    )
    
    inp = _read_json(_input_path(job_id))
    all_warnings: List[str] = []
    processed_results: List[Dict[str, Any]] = []
    
    async with aiohttp.ClientSession() as session:
        for idx, audio_item in enumerate(audio_items):
            audio_url = audio_item.audio_url or audio_item.source_audio_url
            if not audio_url:
                all_warnings.append(f"Result {idx}: No audio URL")
                continue
            
            # Download audio file
            suffix = f"_{idx+1}" if len(audio_items) > 1 else ""
            audio_path = output_dir / f"suno_{audio_item.id}.mp3"
            
            try:
                async with session.get(audio_url) as resp:
                    if resp.status != 200:
                        all_warnings.append(f"Result {idx}: Download failed ({resp.status})")
                        continue
                    audio_path.write_bytes(await resp.read())
            except Exception as e:
                all_warnings.append(f"Result {idx}: Download error - {e}")
                continue
            
            _set_status(job_id, stage=f"processing_{idx+1}/{len(audio_items)}", progress=85 + idx * 5)
            
            # Create separate output dir for each result
            result_output_dir = output_dir / f"result_{idx+1}"
            result_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run post-processing (Demucs + normalize)
            try:
                warnings = await post_process_audio(
                    audio_path=audio_path,
                    output_dir=result_output_dir,
                    job_dir=jd,
                    job_id=job_id,
                    inp=inp,
                    set_status=_set_status,
                )
                all_warnings.extend(warnings)
                
                # Collect result paths
                result_info = {
                    "index": idx + 1,
                    "id": audio_item.id,
                    "title": audio_item.title,
                    "duration": audio_item.duration,
                    "mixed_path": str(audio_path),
                }
                
                # Find processed files
                vocals = list(result_output_dir.rglob("vocals.wav"))
                mr = list(result_output_dir.rglob("no_vocals.wav"))
                norm_vocals = list(result_output_dir.rglob("vocals_*hz*.wav"))
                
                if vocals:
                    result_info["vocals_path"] = str(vocals[0])
                if mr:
                    result_info["mr_path"] = str(mr[0])
                if norm_vocals:
                    result_info["vocals_norm_path"] = str(norm_vocals[0])
                
                processed_results.append(result_info)
                
            except Exception as e:
                all_warnings.append(f"Result {idx}: Post-processing error - {e}")
                continue
    
    if not processed_results:
        _set_status(job_id, status="failed", stage="all_processing_failed", error="All results failed", warnings=all_warnings)
        return {"status": "error", "reason": "all results failed to process"}
    
    # Use first result as primary
    primary = processed_results[0]
    _set_status(
        job_id,
        status="succeeded",
        stage="done",
        progress=100,
        mixed_path=primary.get("mixed_path"),
        vocals_path=primary.get("vocals_path"),
        mr_path=primary.get("mr_path"),
        vocals_norm_path=primary.get("vocals_norm_path"),
        all_results=processed_results,
        warnings=all_warnings or None,
        finished_at=_now(),
    )
    
    logger.info(f"[{job_id}] callback processed successfully - {len(processed_results)} results")
    return {"status": "success", "job_id": job_id, "results_count": len(processed_results)}

@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    """작업 상태 조회"""
    sp = _status_path(job_id)
    if not sp.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return _read_json(sp)

@app.get("/jobs/{job_id}/files/{file_type}")
def download(job_id: str, file_type: str):
    """
    첫 번째 결과 파일 다운로드.
    
    file_type:
    - output: 메인 오디오 파일 (mixed)
    - vocal: 보컬 트랙 (Demucs 분리 후)
    - mr: MR 트랙 (Demucs 분리 후)
    
    Suno 두 번째 결과는 /jobs/{job_id}/results/2/files/{file_type} 사용
    """
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


@app.get("/jobs/{job_id}/results/{index}/files/{file_type}")
def download_by_index(job_id: str, index: int, file_type: str):
    """
    특정 결과 파일 다운로드 (Suno는 2개 결과 생성).
    
    index: 결과 번호 (1 또는 2)
    file_type: output | vocal | mr
    """
    jd = _job_dir(job_id)
    sp = _status_path(job_id)
    if not jd.exists() or not sp.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    st = _read_json(sp)
    all_results = st.get("all_results", [])
    
    if not all_results:
        raise HTTPException(status_code=404, detail="No results available")
    
    if index < 1 or index > len(all_results):
        raise HTTPException(status_code=404, detail=f"Result index {index} not found (available: 1-{len(all_results)})")
    
    result = all_results[index - 1]
    
    path_key = {
        "output": "mixed_path",
        "vocal": "vocals_norm_path",
        "mr": "mr_path",
    }.get(file_type)
    
    if not path_key:
        raise HTTPException(status_code=400, detail="Invalid file_type (output|vocal|mr)")
    
    # For vocal, fall back to vocals_path if vocals_norm_path not available
    p = result.get(path_key)
    if file_type == "vocal" and not p:
        p = result.get("vocals_path")
    
    if not p:
        raise HTTPException(status_code=404, detail=f"{file_type} not ready for result {index}")
    
    pp = Path(p)
    if not pp.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {pp.name}")
    
    return FileResponse(str(pp), media_type=_media_type_for_path(pp), filename=f"{file_type}_{index}_{job_id}{pp.suffix}")

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
