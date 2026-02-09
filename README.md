# Music Generation API

Multi-provider 음악 생성 API 서버 (YuE 로컬 + Suno API)

## Quick Start

```bash
# Docker build
docker build -t music-api .

# Run with Suno API token
docker run -d -p 8000:8000 \
  -e SUNO_API_TOKEN=your_token_here \
  music-api
```

## Providers

| Provider | Type | 장점 |
|----------|------|------|
| `yue` | Local GPU | 완전한 제어, 비용 없음, 오프라인 가능 |
| `suno` | External API | 빠름, GPU 불필요, 다양한 스타일 |

## API Endpoints

### 음악 생성
```bash
# YuE (local)
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "yue",
    "genre": "K-Pop, Dance",
    "lyrics": "[verse]\n가사 내용"
  }'

# Suno (API)
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "suno",
    "prompt": "calm piano melody",
    "style": "Classical",
    "title": "Test Song",
    "instrumental": true
  }'
```

### 음악 연장 (Suno only)
```bash
curl -X POST http://localhost:8000/jobs/extend \
  -H "Content-Type: application/json" \
  -d '{
    "audio_id": "suno-audio-id",
    "continue_at": 60
  }'
```

### 상태 확인 & 다운로드
```bash
# 상태 확인
curl http://localhost:8000/jobs/{job_id}

# 파일 다운로드
curl http://localhost:8000/jobs/{job_id}/files/output  # 메인 오디오
curl http://localhost:8000/jobs/{job_id}/files/vocal   # 보컬 (YuE only)
curl http://localhost:8000/jobs/{job_id}/files/mr      # MR (YuE only)

# Suno 레코드 정보
curl http://localhost:8000/jobs/suno/record-info
```

## Environment Variables

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `SUNO_API_TOKEN` | (필수) | Suno API Bearer 토큰 |
| `SUNO_API_BASE` | `https://api.sunoapi.org/api/v1` | Suno API 베이스 URL |
| `YUE_PATH` | `/opt/YuE/inference` | YuE 추론 경로 |
| `JOBS_DIR` | `/tmp/yue_jobs` | 작업 저장 디렉토리 |

## Response Example

```json
{
  "job_id": "abc12345",
  "status": "queued",
  "status_url": "/jobs/abc12345",
  "provider": "suno"
}
```
