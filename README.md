<p align="center">
  <img src="assets/banner.png" alt="LiteVLM Banner" width="800">
</p>

# LiteVLM Queue API

A production-style image and chat inference stack with:

- BullMQ queue buffering
- Redis-backed job coordination
- FastAPI model runtime
- Bearer-token protected public API

The stack is built for burst traffic. The API accepts requests immediately, then the worker processes jobs at a controlled rate.

## Components

- `queue-api` (`litevlm-queue-api`): Public HTTP API, auth check, request validation, queue producer.
- `queue-worker` (`litevlm-queue-worker`): BullMQ worker, rate limiter, calls model API.
- `model-api` (`litevlm-model-api`): Python inference service (SmolVLM + Qwen), lazy model loading.
- `redis` (`litevlm-redis`): Queue state, locks, events.

```mermaid
graph LR
    Client([Client]) -->|HTTP POST| API[queue-api]
    API -->|BullMQ| Redis[(Redis)]
    Worker[queue-worker] -->|BullMQ| Redis
    Worker -->|HTTP POST| Model[model-api]
```

## Request Flow

1. Client sends request to `queue-api`.
2. `queue-api` validates token and payload.
3. Valid request is enqueued to Redis.
4. `queue-worker` picks jobs in FIFO order with rate limiting.
5. Worker calls `model-api` and returns response.
6. `queue-api` waits for completion (up to `JOB_TIMEOUT_MS`) and returns result.

```mermaid
sequenceDiagram
    participant Client
    participant QueueAPI as queue-api
    participant Redis as redis
    participant Worker as queue-worker
    participant ModelAPI as model-api

    Client->>QueueAPI: POST /generate
    QueueAPI->>Redis: Enqueue Job
    QueueAPI-->>Client: (Waits for result)
    
    Worker->>Redis: Poll for Jobs
    Redis-->>Worker: Job payload
    
    Worker->>ModelAPI: POST /generate
    ModelAPI-->>Worker: Model Inference Result
    
    Worker->>Redis: Mark Job Completed
    Redis-->>QueueAPI: Job Completion Event
    QueueAPI-->>Client: Return HTTP 200 with Result
```

## Supported Models

- `smolvlm-256m` -> `HuggingFaceTB/SmolVLM-256M-Instruct` (vision + text)
- `qwen-1.5b` -> `Qwen/Qwen2.5-1.5B-Instruct` (text)

Model alias names are part of the API contract. Keep them stable unless you version your clients.

## Default Server Requirements

These are practical defaults for CPU deployment.

### Minimum profile (functional)

- 4 vCPU
- 16 GB RAM
- 40 GB SSD free space
- Ubuntu 22.04+ or Debian 12+
- Docker Engine 24+ and Docker Compose v2+
- Stable outbound network to `huggingface.co`

### Recommended profile (stable under real traffic)

- 8 vCPU
- 32 GB RAM
- 80 GB SSD free space
- Same OS/runtime as above

### Notes

- First model load downloads weights and can take time.
- If `SINGLE_ACTIVE_MODEL=false`, both models may stay resident; memory usage increases.
- Keep at least 25-30 GB free disk for model cache and container layers.
- Redis warning fix on Linux hosts:

```bash
sudo sysctl vm.overcommit_memory=1
```

Persist it in `/etc/sysctl.conf` for reboot survival.

## Repository Layout

```text
.
├── app/
│   └── main.py                 # FastAPI model runtime
├── gateway/
│   ├── auth-tokens.json        # Allowed bearer tokens
│   ├── src/
│   │   ├── api.js              # Public queue API
│   │   ├── worker.js           # BullMQ worker
│   │   ├── queue.js            # Queue + Redis wiring
│   │   └── config.js           # Env parsing/validation
│   └── package.json
├── docker-compose.yml
├── Dockerfile                  # model-api image
└── .env.example
```

## Quick Start

1. Copy environment file:

```bash
cp .env.example .env
```

2. Set at least one real token in `gateway/auth-tokens.json`:

```json
{
  "tokens": [
    "replace-with-strong-token"
  ]
}
```

3. (Optional) set `HF_TOKEN` in `.env` if you use gated Hugging Face models.

4. Start stack:

```bash
docker compose up --build
```

5. Public API base URL:

```text
http://localhost:8000
```

## Public Endpoints (`queue-api`)

All public endpoints require:

```text
Authorization: Bearer <token>
```

### `GET /health`

Returns queue status and basic config.

Example:

```bash
curl -sS http://localhost:8000/health \
  -H "Authorization: Bearer replace-with-strong-token"
```

### `POST /generate`

- Content type: `multipart/form-data`
- Required fields: `image`, `prompt`
- Optional fields: `model`, `max_new_tokens`

Example:

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Authorization: Bearer replace-with-strong-token" \
  -F "image=@/absolute/path/to/image.jpg" \
  -F "prompt=Describe this image in one paragraph." \
  -F "model=smolvlm-256m" \
  -F "max_new_tokens=96"
```

### `POST /chat`

- Content type: `application/json`
- Required: non-empty `messages`
- Optional: `model`, `max_new_tokens`, `max_tokens`

Example:

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Authorization: Bearer replace-with-strong-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-1.5b",
    "messages": [
      {"role": "user", "content": "Explain queue-based APIs in one paragraph."}
    ],
    "max_new_tokens": 128
  }'
```

### `POST /v1/chat/completions`

OpenAI-style payload and response shape.

Example:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Authorization: Bearer replace-with-strong-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-1.5b",
    "messages": [
      {"role": "system", "content": "Reply in Turkish."},
      {"role": "user", "content": "Rate limiting neden gereklidir?"}
    ],
    "max_tokens": 120
  }'
```

## Internal Endpoints (`model-api`)

These are internal by design (service-to-service):

- `GET /health`
- `GET /models`
- `POST /generate`
- `POST /chat`
- `POST /v1/chat/completions`

`queue-api` does not currently proxy `/models`.

## Model Selection Rules

- Default alias: `DEFAULT_MODEL_ALIAS`.
- Vision input requires `smolvlm-256m`.
- `qwen-1.5b` rejects image content.
- Models load lazily on first request.
- Idle models unload after `MODEL_IDLE_UNLOAD_SECONDS`.
- Each successful use refreshes the model timer.
- If `SINGLE_ACTIVE_MODEL=true`, loading one model unloads the others.

## Queue and Throughput Rules

- Queue name default: `litevlm_requests`.
- Rate limit default: 10 jobs/minute (`MAX_REQUESTS_PER_MINUTE=10`).
- Worker concurrency default: 1 (`WORKER_CONCURRENCY=1`).
- Queue is FIFO.
- API wait timeout default: 300000 ms (`JOB_TIMEOUT_MS`).

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Public port for `queue-api`. |
| `REDIS_URL` | `redis://redis:6379` | Redis connection string. |
| `QUEUE_NAME` | `litevlm_requests` | BullMQ queue key namespace. |
| `MODEL_API_BASE_URL` | `http://model-api:8000` | Internal URL used by worker. |
| `MAX_REQUESTS_PER_MINUTE` | `10` | Worker throughput cap per 60s. |
| `WORKER_CONCURRENCY` | `1` | Parallel jobs per worker process. |
| `WORKER_LOCK_DURATION_MS` | `300000` | BullMQ active job lock TTL. |
| `WORKER_STALLED_INTERVAL_MS` | `30000` | Stalled-job check interval. |
| `WORKER_MAX_STALLED_COUNT` | `1` | Stalled retries before fail. |
| `JOB_TIMEOUT_MS` | `300000` | Queue API wait timeout for job completion. |
| `UPLOAD_LIMIT_MB` | `10` | Max multipart image size for `/generate`. |
| `JSON_BODY_LIMIT_MB` | `25` | Max JSON body size for chat endpoints. |
| `AUTH_TOKENS_FILE` | `/app/auth-tokens.json` | Allowed bearer token file path. |
| `HF_TOKEN` | empty | Hugging Face token (needed for gated models). |
| `DEVICE` | `cpu` | Inference device (`cpu` or `cuda`). |
| `DEFAULT_MAX_NEW_TOKENS` | `64` | Default generation length. |
| `MAX_NEW_TOKENS_LIMIT` | `256` | Hard max token limit for requests. |
| `LOG_LEVEL` | `INFO` | Log level for model API. |
| `DEFAULT_MODEL_ALIAS` | `smolvlm-256m` | Default model when request omits model. |
| `SMOLVLM_MODEL_ID` | `HuggingFaceTB/SmolVLM-256M-Instruct` | Hugging Face model id for vision alias. |
| `QWEN_MODEL_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face model id for text alias. |
| `SINGLE_ACTIVE_MODEL` | `false` | Keep only one loaded model at a time. |
| `MODEL_IDLE_UNLOAD_SECONDS` | `3600` | Idle model unload threshold. |
| `MODEL_CLEANUP_INTERVAL_SECONDS` | `60` | Idle check interval. |

## Operations

### Check service status

```bash
docker compose ps
```

### Tail logs

```bash
docker compose logs -f queue-api queue-worker model-api redis
```

### Restart a single service

```bash
docker compose restart queue-worker
```

### Rebuild after code changes

```bash
docker compose up --build -d
```

## Troubleshooting

### `503 Worker could not reach model API: fetch failed`

- `model-api` is down, restarting, or unreachable from worker.
- Check:

```bash
docker compose ps
docker compose logs --tail=200 model-api queue-worker
```

### `503 Model is not loaded`

- Model load failed previously.
- Check model-api logs for the first root error (often Hugging Face access or download issue).

### `403 ... gated repo`

- The configured model requires Hugging Face access.
- Set a valid `HF_TOKEN` and ensure your account has access to that repo.

### BullMQ lock errors (`could not renew lock`, `Missing lock for job`)

- Job runtime exceeded lock assumptions.
- Increase:
  - `WORKER_LOCK_DURATION_MS`
  - `WORKER_STALLED_INTERVAL_MS`

### Redis overcommit warning

- Set host kernel parameter:

```bash
sudo sysctl vm.overcommit_memory=1
```

## Security Notes

- Keep `gateway/auth-tokens.json` out of public repositories.
- Rotate bearer tokens regularly.
- Do not expose Redis publicly.
- Keep `model-api` internal unless you add dedicated auth/rate limiting.

## Naming Conventions

Compose/container and queue naming now follow LiteVLM naming:

- `litevlm-queue-api`
- `litevlm-queue-worker`
- `litevlm-model-api`
- `litevlm-redis`
- `litevlm_requests`
