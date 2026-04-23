const IORedis = require("ioredis");
const { Queue, QueueEvents } = require("bullmq");
const { REDIS_URL, QUEUE_NAME } = require("./config");

function createRedisConnection() {
  return new IORedis(REDIS_URL, {
    maxRetriesPerRequest: null,
    enableReadyCheck: true,
  });
}

const queueConnection = createRedisConnection();
const queueEventsConnection = createRedisConnection();

const requestQueue = new Queue(QUEUE_NAME, {
  connection: queueConnection,
  defaultJobOptions: {
    removeOnComplete: 200,
    removeOnFail: 200,
  },
});

const queueEvents = new QueueEvents(QUEUE_NAME, {
  connection: queueEventsConnection,
});

queueEvents.on("error", (error) => {
  console.error("QueueEvents error:", error);
});

async function closeQueueResources() {
  await queueEvents.close();
  await requestQueue.close();
  await queueConnection.quit();
  await queueEventsConnection.quit();
}

module.exports = {
  createRedisConnection,
  requestQueue,
  queueEvents,
  closeQueueResources,
};
