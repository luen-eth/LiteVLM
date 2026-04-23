const dotenv = require("dotenv");

dotenv.config();

function readInt(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined || raw === "") {
    return fallback;
  }

  const value = Number.parseInt(raw, 10);
  if (!Number.isFinite(value)) {
    throw new Error(`Environment variable ${name} must be an integer`);
  }
  return value;
}

function readString(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined || raw === "") {
    return fallback;
  }
  return raw;
}

const PORT = readInt("PORT", 8000);
const REDIS_URL = readString("REDIS_URL", "redis://redis:6379");
const QUEUE_NAME = readString("QUEUE_NAME", "litevlm_requests");
const MODEL_API_BASE_URL = readString("MODEL_API_BASE_URL", "http://model-api:8000").replace(/\/$/, "");
const MAX_REQUESTS_PER_MINUTE = readInt("MAX_REQUESTS_PER_MINUTE", 10);
const JOB_TIMEOUT_MS = readInt("JOB_TIMEOUT_MS", 300000);
const WORKER_CONCURRENCY = readInt("WORKER_CONCURRENCY", 1);
const WORKER_LOCK_DURATION_MS = readInt("WORKER_LOCK_DURATION_MS", 300000);
const WORKER_STALLED_INTERVAL_MS = readInt("WORKER_STALLED_INTERVAL_MS", 30000);
const WORKER_MAX_STALLED_COUNT = readInt("WORKER_MAX_STALLED_COUNT", 1);
const UPLOAD_LIMIT_MB = readInt("UPLOAD_LIMIT_MB", 10);
const JSON_BODY_LIMIT_MB = readInt("JSON_BODY_LIMIT_MB", 25);
const AUTH_TOKENS_FILE = readString("AUTH_TOKENS_FILE", "/app/auth-tokens.json");

if (MAX_REQUESTS_PER_MINUTE < 1) {
  throw new Error("MAX_REQUESTS_PER_MINUTE must be greater than 0");
}

if (WORKER_CONCURRENCY < 1) {
  throw new Error("WORKER_CONCURRENCY must be greater than 0");
}

if (WORKER_LOCK_DURATION_MS < 1000) {
  throw new Error("WORKER_LOCK_DURATION_MS must be at least 1000");
}

if (WORKER_STALLED_INTERVAL_MS < 1000) {
  throw new Error("WORKER_STALLED_INTERVAL_MS must be at least 1000");
}

if (WORKER_MAX_STALLED_COUNT < 0) {
  throw new Error("WORKER_MAX_STALLED_COUNT must be 0 or greater");
}

module.exports = {
  PORT,
  REDIS_URL,
  QUEUE_NAME,
  MODEL_API_BASE_URL,
  MAX_REQUESTS_PER_MINUTE,
  JOB_TIMEOUT_MS,
  WORKER_CONCURRENCY,
  WORKER_LOCK_DURATION_MS,
  WORKER_STALLED_INTERVAL_MS,
  WORKER_MAX_STALLED_COUNT,
  UPLOAD_LIMIT_MB,
  JSON_BODY_LIMIT_MB,
  AUTH_TOKENS_FILE,
};
