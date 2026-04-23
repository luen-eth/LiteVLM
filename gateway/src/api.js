const fs = require("fs");
const express = require("express");
const multer = require("multer");

const {
  PORT,
  QUEUE_NAME,
  MAX_REQUESTS_PER_MINUTE,
  MODEL_API_BASE_URL,
  JOB_TIMEOUT_MS,
  UPLOAD_LIMIT_MB,
  JSON_BODY_LIMIT_MB,
  AUTH_TOKENS_FILE,
} = require("./config");
const { requestQueue, queueEvents, closeQueueResources } = require("./queue");

const app = express();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: UPLOAD_LIMIT_MB * 1024 * 1024,
  },
});

app.use(express.json({ limit: `${JSON_BODY_LIMIT_MB}mb` }));

function loadAuthTokens(filePath) {
  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch (error) {
    throw new Error(`Failed to read auth tokens file at ${filePath}: ${error.message}`);
  }

  const tokenList = Array.isArray(parsed) ? parsed : parsed.tokens;
  if (!Array.isArray(tokenList) || tokenList.length === 0) {
    throw new Error(`Auth tokens file ${filePath} must include a non-empty token list`);
  }

  const normalized = tokenList
    .map((token) => String(token).trim())
    .filter((token) => token.length > 0);

  if (normalized.length === 0) {
    throw new Error(`Auth tokens file ${filePath} must include at least one non-empty token`);
  }

  return new Set(normalized);
}

const allowedTokens = loadAuthTokens(AUTH_TOKENS_FILE);
console.log(`Loaded ${allowedTokens.size} auth token(s) from ${AUTH_TOKENS_FILE}`);

function requireAuth(req, res, next) {
  const authHeader = req.headers.authorization || "";

  if (!authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ detail: "Missing Bearer token" });
  }

  const token = authHeader.slice("Bearer ".length).trim();
  if (!token || !allowedTokens.has(token)) {
    return res.status(403).json({ detail: "Invalid token" });
  }

  return next();
}

function parseInteger(value, fieldName) {
  if (value === undefined || value === null || value === "") {
    return undefined;
  }

  const parsed = Number.parseInt(String(value), 10);
  if (!Number.isFinite(parsed)) {
    throw new Error(`${fieldName} must be an integer`);
  }
  return parsed;
}

function normalizeWorkerError(error) {
  if (!error) {
    return null;
  }

  const message = error.message || String(error);
  try {
    const parsed = JSON.parse(message);
    if (parsed && typeof parsed.status === "number") {
      return parsed;
    }
  } catch {
    // ignore parse errors and return null below
  }

  return null;
}

async function enqueueAndWait(jobName, payload) {
  const job = await requestQueue.add(jobName, payload);
  return job.waitUntilFinished(queueEvents, JOB_TIMEOUT_MS);
}

app.use(requireAuth);

app.get("/health", async (_req, res) => {
  const queueCounts = await requestQueue.getJobCounts(
    "waiting",
    "active",
    "completed",
    "failed",
    "delayed"
  );

  res.json({
    ok: true,
    auth_enabled: true,
    queue_name: QUEUE_NAME,
    max_requests_per_minute: MAX_REQUESTS_PER_MINUTE,
    model_api_base_url: MODEL_API_BASE_URL,
    queue_counts: queueCounts,
  });
});

app.post("/generate", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ detail: "image is required" });
    }

    const prompt = (req.body.prompt || "").trim();
    if (!prompt) {
      return res.status(400).json({ detail: "prompt cannot be empty" });
    }

    const maxNewTokens = parseInteger(req.body.max_new_tokens, "max_new_tokens");
    const model = typeof req.body.model === "string" && req.body.model.trim() ? req.body.model.trim() : undefined;

    const response = await enqueueAndWait("generate", {
      image_base64: req.file.buffer.toString("base64"),
      image_mime_type: req.file.mimetype || "image/jpeg",
      image_filename: req.file.originalname || "upload.jpg",
      prompt,
      max_new_tokens: maxNewTokens,
      model,
    });

    return res.status(200).json(response);
  } catch (error) {
    const workerError = normalizeWorkerError(error);
    if (workerError) {
      return res.status(workerError.status).json(workerError.body || { detail: "Worker error" });
    }

    if (String(error.message || "").toLowerCase().includes("timed out")) {
      return res.status(504).json({ detail: "Request timed out in queue" });
    }

    return res.status(500).json({ detail: error.message || "Unexpected queue error" });
  }
});

app.post("/chat", async (req, res) => {
  try {
    if (!req.body || !Array.isArray(req.body.messages) || req.body.messages.length === 0) {
      return res.status(400).json({ detail: "messages must be a non-empty array" });
    }

    const response = await enqueueAndWait("chat", {
      request: req.body,
    });

    return res.status(200).json(response);
  } catch (error) {
    const workerError = normalizeWorkerError(error);
    if (workerError) {
      return res.status(workerError.status).json(workerError.body || { detail: "Worker error" });
    }

    if (String(error.message || "").toLowerCase().includes("timed out")) {
      return res.status(504).json({ detail: "Request timed out in queue" });
    }

    return res.status(500).json({ detail: error.message || "Unexpected queue error" });
  }
});

app.post("/v1/chat/completions", async (req, res) => {
  try {
    if (!req.body || !Array.isArray(req.body.messages) || req.body.messages.length === 0) {
      return res.status(400).json({ detail: "messages must be a non-empty array" });
    }

    const response = await enqueueAndWait("chat_completions", {
      request: req.body,
    });

    return res.status(200).json(response);
  } catch (error) {
    const workerError = normalizeWorkerError(error);
    if (workerError) {
      return res.status(workerError.status).json(workerError.body || { detail: "Worker error" });
    }

    if (String(error.message || "").toLowerCase().includes("timed out")) {
      return res.status(504).json({ detail: "Request timed out in queue" });
    }

    return res.status(500).json({ detail: error.message || "Unexpected queue error" });
  }
});

let server;

async function start() {
  await queueEvents.waitUntilReady();

  server = app.listen(PORT, () => {
    console.log(`Queue API is listening on port ${PORT}`);
  });
}

async function shutdown(signal) {
  console.log(`Received ${signal}, shutting down...`);

  if (server) {
    await new Promise((resolve) => server.close(resolve));
  }

  await closeQueueResources();
  process.exit(0);
}

process.on("SIGTERM", () => {
  void shutdown("SIGTERM");
});

process.on("SIGINT", () => {
  void shutdown("SIGINT");
});

start().catch((error) => {
  console.error("Failed to start Queue API:", error);
  process.exit(1);
});
