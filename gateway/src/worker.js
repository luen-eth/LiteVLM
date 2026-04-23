const { Blob } = require("buffer");
const { Worker } = require("bullmq");

const {
  QUEUE_NAME,
  MODEL_API_BASE_URL,
  MAX_REQUESTS_PER_MINUTE,
  WORKER_CONCURRENCY,
  WORKER_LOCK_DURATION_MS,
  WORKER_STALLED_INTERVAL_MS,
  WORKER_MAX_STALLED_COUNT,
} = require("./config");
const { createRedisConnection } = require("./queue");

const connection = createRedisConnection();

async function parseBody(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

async function callModelJson(path, body) {
  let response;
  try {
    response = await fetch(`${MODEL_API_BASE_URL}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
  } catch (error) {
    throw new Error(
      JSON.stringify({
        status: 503,
        body: { detail: `Worker could not reach model API: ${error.message}` },
      })
    );
  }

  const parsedBody = await parseBody(response);
  if (!response.ok) {
    throw new Error(
      JSON.stringify({
        status: response.status,
        body: parsedBody || { detail: "Model API returned an error" },
      })
    );
  }

  return parsedBody;
}

async function callModelGenerate(data) {
  let response;

  const imageBytes = Buffer.from(data.image_base64, "base64");
  const form = new FormData();
  form.append(
    "image",
    new Blob([imageBytes], { type: data.image_mime_type || "image/jpeg" }),
    data.image_filename || "upload.jpg"
  );
  form.append("prompt", data.prompt);
  if (data.max_new_tokens !== undefined && data.max_new_tokens !== null) {
    form.append("max_new_tokens", String(data.max_new_tokens));
  }
  if (data.model) {
    form.append("model", String(data.model));
  }

  try {
    response = await fetch(`${MODEL_API_BASE_URL}/generate`, {
      method: "POST",
      body: form,
    });
  } catch (error) {
    throw new Error(
      JSON.stringify({
        status: 503,
        body: { detail: `Worker could not reach model API: ${error.message}` },
      })
    );
  }

  const parsedBody = await parseBody(response);
  if (!response.ok) {
    throw new Error(
      JSON.stringify({
        status: response.status,
        body: parsedBody || { detail: "Model API returned an error" },
      })
    );
  }

  return parsedBody;
}

const worker = new Worker(
  QUEUE_NAME,
  async (job) => {
    if (job.name === "generate") {
      return callModelGenerate(job.data);
    }

    if (job.name === "chat") {
      return callModelJson("/chat", job.data.request);
    }

    if (job.name === "chat_completions") {
      return callModelJson("/v1/chat/completions", job.data.request);
    }

    throw new Error(
      JSON.stringify({
        status: 400,
        body: { detail: `Unsupported job type: ${job.name}` },
      })
    );
  },
  {
    connection,
    concurrency: WORKER_CONCURRENCY,
    lockDuration: WORKER_LOCK_DURATION_MS,
    stalledInterval: WORKER_STALLED_INTERVAL_MS,
    maxStalledCount: WORKER_MAX_STALLED_COUNT,
    limiter: {
      max: MAX_REQUESTS_PER_MINUTE,
      duration: 60_000,
    },
  }
);

worker.on("ready", () => {
  console.log(
    `Queue worker is ready. queue=${QUEUE_NAME} concurrency=${WORKER_CONCURRENCY} limit=${MAX_REQUESTS_PER_MINUTE}/min lock=${WORKER_LOCK_DURATION_MS}ms stalledInterval=${WORKER_STALLED_INTERVAL_MS}ms`
  );
});

worker.on("completed", (job) => {
  console.log(`Job completed: id=${job.id} name=${job.name}`);
});

worker.on("failed", (job, error) => {
  const id = job ? job.id : "unknown";
  const name = job ? job.name : "unknown";
  console.error(`Job failed: id=${id} name=${name}`, error);
});

async function shutdown(signal) {
  console.log(`Received ${signal}, shutting down worker...`);
  await worker.close();
  await connection.quit();
  process.exit(0);
}

process.on("SIGTERM", () => {
  void shutdown("SIGTERM");
});

process.on("SIGINT", () => {
  void shutdown("SIGINT");
});
