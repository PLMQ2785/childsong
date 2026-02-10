# 🎵 MakeSong 프로젝트 회고 및 정리

**프로젝트명:** childsong (Music Generation API)  
**버전:** 3.1.0  
**작성일:** 2026-02-09

---

## 📋 프로젝트 개요

AI 기반 음악 생성 API 서버. 두 가지 Provider를 지원:

| Provider | 방식 | 특징 |
|----------|------|------|
| **YuE** | 로컬 GPU | 완전 제어, 오프라인 가능 |
| **Suno** | 외부 API | 빠름, 다양한 스타일 |

---

## 📁 프로젝트 구조

```
childsong/
├── api_server.py           # FastAPI 메인 서버 (763줄)
├── providers/
│   ├── suno_provider.py    # Suno API 연동
│   ├── yue_provider.py     # YuE 로컬 생성
│   └── audio_utils.py      # Demucs 후처리
├── Dockerfile              # 컨테이너 이미지
├── docker-compose.yml      # 프로덕션 설정
├── docker-compose-test.yml # 테스트 설정
├── .env                    # 환경변수
├── API_SPEC.md             # API 명세서
└── README.md               # 사용법
```

---

## 🔧 핵심 기술 스택

| 분류 | 기술 |
|------|------|
| 웹 프레임워크 | FastAPI |
| 비동기 | asyncio, aiohttp |
| 오디오 처리 | Demucs, ffmpeg |
| 컨테이너 | Docker, docker-compose |
| 베이스 이미지 | runpod/pytorch (CUDA 12.8.1) |

---

## 🎯 구현한 주요 기능

### 1. 이중 큐 시스템

```python
yue_queue   # YuE 순차 처리 (GPU)
suno_queue  # Suno 동시 처리 (Rate Limit)
```

**목적:** YuE는 GPU 자원 제한으로 순차 처리, Suno는 외부 API라 동시 처리 가능

### 2. Rate Limiter (Sliding Window)

```python
class RateLimiter:
    def __init__(self, max_requests=10, window_seconds=60):
        self.timestamps = []  # 요청 시간 기록
    
    async def acquire(self):
        # 1분 경과한 기록 제거
        # 10개 미만 → 허용
        # 10개 이상 → 대기
```

**핵심:** 분당 10회 Suno API 호출 제한

### 3. 콜백 기반 비동기 처리

```
POST /jobs → Suno API 호출 → 상태: waiting_callback
                    ↓
            Suno가 음악 생성 (3-5분)
                    ↓
            POST /jobs/callback ← Suno 결과 전송
                    ↓
            다운로드 → Demucs → 상태: succeeded
```

### 4. 다중 결과 처리

Suno는 요청당 2개 결과 생성:
- `result_1/` - 첫 번째 (기본)
- `result_2/` - 두 번째

엔드포인트: `/jobs/{id}/results/{index}/files/{type}`

---

## 💡 배운 점

### 1. asyncio 동시성

```python
# 비동기 태스크 생성 (블로킹 없이)
asyncio.create_task(_process_suno_job(job_id))

# 세마포어 대신 Rate Limiter 사용
await suno_rate_limiter.acquire()
```

### 2. Sliding Window 알고리즘

```python
# 오래된 타임스탬프 제거
self.timestamps = [t for t in self.timestamps if now - t < 60]

# 대기 시간 계산
wait_time = window - (now - timestamps[0])
```

### 3. FastAPI 구조화

- Pydantic 모델로 요청/응답 정의
- `on_startup`/`on_shutdown` 이벤트
- 비동기 백그라운드 워커

---

## 🔗 환경변수

| 변수 | 설명 |
|------|------|
| `SUNO_API_TOKEN` | Suno API 인증 토큰 |
| `CALLBACK_BASE_URL` | 콜백 수신 URL (공인 IP) |
| `SUNO_RATE_LIMIT` | 분당 최대 요청 (기본: 10) |

---

## 🚀 실행 방법

```bash
# 테스트 환경
docker-compose -f docker-compose-test.yml up --build -d

# 로그 확인
docker logs -f childsong-test
```

---

## 📊 API 요약

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 헬스체크 |
| POST | `/jobs` | 음악 생성 |
| POST | `/jobs/extend` | 음악 연장 |
| GET | `/jobs/{id}` | 상태 조회 |
| GET | `/jobs/{id}/files/{type}` | 파일 다운로드 |
| POST | `/jobs/callback` | Suno 콜백 |

---

## 🔮 향후 개선 가능

1. Redis 기반 분산 큐
2. WebSocket 실시간 상태 알림
3. S3/GCS 파일 스토리지
4. 사용자 인증 (JWT)
5. 요금 과금 시스템
