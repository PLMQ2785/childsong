# Music Generation API 명세서

**Version:** 3.1.0  
**Base URL:** `http://210.182.182.205:8000`

---

## 1. 헬스체크

### `GET /`

**Response:**
```json
{
  "status": "ok",
  "version": "3.1.0",
  "providers": ["yue", "suno"],
  "yue_queue_size": 0,
  "suno_queue_size": 3,
  "suno_rate_limit": 10
}
```

---

## 2. 음악 생성

### `POST /jobs`

#### YuE (로컬 GPU - 순차 처리)
```json
{
  "provider": "yue",
  "genre": "K-Pop, Dance, Upbeat",
  "lyrics": "[verse]\n가사 내용..."
}
```

#### Suno (외부 API - 동시 처리, 분당 10회)
```json
{
  "provider": "suno",
  "prompt": "[Verse]\n가사 내용",
  "style": "K-pop, children song, happy",
  "title": "노래 제목",
  "model": "V5",
  "instrumental": false
}
```

#### 요청 파라미터

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `provider` | string | ✗ | `yue` (기본) 또는 `suno` |
| `genre` | string | YuE 필수 | 장르 설명 |
| `lyrics` | string | YuE 필수 | 가사 |
| `prompt` | string | Suno 필수* | 프롬프트 (가사 포함 가능) |
| `style` | string | Suno 필수* | 음악 스타일 |
| `title` | string | ✗ | 노래 제목 |
| `model` | string | ✗ | Suno 모델 (기본: `V5`) |
| `instrumental` | bool | ✗ | 인스트루멘탈만 (기본: false) |
| `separate_tracks` | bool | ✗ | Demucs 분리 (기본: true) |
| `normalize_vocals` | bool | ✗ | 보컬 정규화 (기본: true) |
| `vocal_sr` | int | ✗ | 보컬 샘플레이트 (기본: 16000) |

> *Suno: `prompt` 또는 `style` 중 하나 필수

#### Response
```json
{
  "job_id": "abc12345",
  "status": "queued",
  "status_url": "/jobs/abc12345",
  "provider": "suno"
}
```

---

## 3. 음악 연장 (Suno)

### `POST /jobs/extend`

```json
{
  "audio_id": "suno-audio-id",
  "model": "V5",
  "continue_at": 60
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `audio_id` | string | ✓ | Suno 오디오 ID |
| `continue_at` | int | ✗ | 연장 시작 위치 (초) |

---

## 4. 상태 조회

### `GET /jobs/{job_id}`

**Response (진행 중):**
```json
{
  "status": "waiting_callback",
  "stage": "suno_callback_pending",
  "progress": 15,
  "provider": "suno"
}
```

**Response (완료):**
```json
{
  "status": "succeeded",
  "stage": "done",
  "progress": 100,
  "mixed_path": "/tmp/yue_jobs/abc12345/output/suno_xxx.mp3",
  "vocals_path": "...",
  "mr_path": "...",
  "all_results": [
    {"index": 1, "id": "...", "title": "...", "duration": 14.04},
    {"index": 2, "id": "...", "title": "...", "duration": 19.96}
  ]
}
```

#### 상태 값

| status | 설명 |
|--------|------|
| `queued` | 대기 중 |
| `running` | 처리 중 |
| `waiting_callback` | Suno 콜백 대기 |
| `succeeded` | 완료 |
| `failed` | 실패 |

---

## 5. 파일 다운로드

### `GET /jobs/{job_id}/files/{file_type}`
첫 번째 결과 다운로드

### `GET /jobs/{job_id}/results/{index}/files/{file_type}`
특정 결과 다운로드 (Suno: 1 또는 2)

| file_type | 설명 |
|-----------|------|
| `output` | 전체 오디오 (MP3) |
| `vocal` | 보컬 트랙 (WAV) |
| `mr` | MR 트랙 (WAV) |

---

## 6. Suno 레코드 조회

### `GET /jobs/suno/record-info`
Suno 계정의 생성된 음악 목록 조회

---

## 7. 콜백 (내부용)

### `POST /jobs/callback`
Suno API 콜백 수신 (자동 처리)

---

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `SUNO_API_TOKEN` | - | Suno API 토큰 (필수) |
| `CALLBACK_BASE_URL` | - | 콜백 수신 URL |
| `SUNO_RATE_LIMIT` | 10 | 분당 최대 Suno 요청 수 |
| `SUNO_API_BASE` | `https://api.sunoapi.org/api/v1` | Suno API 베이스 |
| `JOBS_DIR` | `/tmp/yue_jobs` | 작업 디렉토리 |

---

## 처리 방식

| Provider | 큐 | 처리 방식 |
|----------|-----|----------|
| YuE | `yue_queue` | 순차 (GPU 사용) |
| Suno | `suno_queue` | 동시 처리 (Rate Limit: 분당 10회) |
