import base64
import binascii
import gc
import io
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_MODEL_ALIAS = os.getenv("DEFAULT_MODEL_ALIAS", "smolvlm-256m")
SMOLVLM_MODEL_ID = os.getenv("SMOLVLM_MODEL_ID", "HuggingFaceTB/SmolVLM-256M-Instruct")
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID") or os.getenv(
    "GEMMA_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"
)
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "64"))
MAX_NEW_TOKENS_LIMIT = int(os.getenv("MAX_NEW_TOKENS_LIMIT", "256"))
MODEL_IDLE_UNLOAD_SECONDS = int(os.getenv("MODEL_IDLE_UNLOAD_SECONDS", "3600"))
MODEL_CLEANUP_INTERVAL_SECONDS = int(os.getenv("MODEL_CLEANUP_INTERVAL_SECONDS", "60"))
SINGLE_ACTIVE_MODEL = _read_bool_env("SINGLE_ACTIVE_MODEL", False)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("model-api")


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    kind: str  # "vision" or "text"


MODEL_SPECS = {
    "smolvlm-256m": ModelSpec(
        alias="smolvlm-256m",
        model_id=SMOLVLM_MODEL_ID,
        kind="vision",
    ),
    "qwen-1.5b": ModelSpec(
        alias="qwen-1.5b",
        model_id=QWEN_MODEL_ID,
        kind="text",
    ),
}

if DEFAULT_MODEL_ALIAS not in MODEL_SPECS:
    raise RuntimeError(
        f"DEFAULT_MODEL_ALIAS must be one of: {', '.join(sorted(MODEL_SPECS.keys()))}"
    )

if MODEL_IDLE_UNLOAD_SECONDS < 1:
    raise RuntimeError("MODEL_IDLE_UNLOAD_SECONDS must be >= 1")

if MODEL_CLEANUP_INTERVAL_SECONDS < 1:
    raise RuntimeError("MODEL_CLEANUP_INTERVAL_SECONDS must be >= 1")


class ImageURL(BaseModel):
    url: str


class ChatContentItem(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageURL | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[ChatContentItem]


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = None
    max_new_tokens: int | None = None


class VisionRuntime:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._lock = threading.Lock()
        self.processor: Any | None = None
        self.model: Any | None = None

    def is_loaded(self) -> bool:
        return self.processor is not None and self.model is not None

    def load(self) -> None:
        if self.is_loaded():
            return

        self.processor = AutoProcessor.from_pretrained(self.spec.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(self.spec.model_id)
        self.model.to(DEVICE)
        self.model.eval()
        logger.info("Loaded vision model: alias=%s model=%s", self.spec.alias, self.spec.model_id)

    def unload(self) -> None:
        with self._lock:
            if self.model is not None:
                del self.model
            if self.processor is not None:
                del self.processor
            self.model = None
            self.processor = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        logger.info("Unloaded vision model: alias=%s", self.spec.alias)

    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        max_new_tokens: int,
    ) -> str:
        if self.processor is None or self.model is None:
            raise RuntimeError("Model is not loaded")

        rendered_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        with self._lock:
            if images:
                inputs = self.processor(text=rendered_prompt, images=images, return_tensors="pt")
            else:
                inputs = self.processor(text=rendered_prompt, return_tensors="pt")

            inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        output = generated_texts[0].strip() if generated_texts else ""
        if "Assistant:" in output:
            output = output.split("Assistant:", 1)[1].strip()
        return output


class TextRuntime:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._lock = threading.Lock()
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def is_loaded(self) -> bool:
        return self.tokenizer is not None and self.model is not None

    def load(self) -> None:
        if self.is_loaded():
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id, token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(self.spec.model_id, token=HF_TOKEN)
        self.model.to(DEVICE)
        self.model.eval()
        logger.info("Loaded text model: alias=%s model=%s", self.spec.alias, self.spec.model_id)

    def unload(self) -> None:
        with self._lock:
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer
            self.model = None
            self.tokenizer = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        logger.info("Unloaded text model: alias=%s", self.spec.alias)

    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        max_new_tokens: int,
    ) -> str:
        if images:
            raise RuntimeError("Selected model does not support image input")
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model is not loaded")

        text_messages: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", "")).strip()
            content = message.get("content")
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    item_type = item.get("type")
                    if item_type != "text":
                        raise RuntimeError("Selected model does not support image input")
                    part = str(item.get("text", "")).strip()
                    if part:
                        text_parts.append(part)
                text = "\n".join(text_parts).strip()
            else:
                text = ""

            if not role or not text:
                continue
            text_messages.append({"role": role, "content": text})

        if not text_messages:
            raise RuntimeError("Text request is empty")

        rendered_prompt = self.tokenizer.apply_chat_template(
            text_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        with self._lock:
            model_inputs = self.tokenizer([rendered_prompt], return_tensors="pt")
            model_inputs = {
                k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in model_inputs.items()
            }
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                )

        input_len = model_inputs["input_ids"].shape[1]
        completion_ids = generated_ids[:, input_len:]
        output = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return output[0].strip() if output else ""


@dataclass
class ModelEntry:
    spec: ModelSpec
    runtime: VisionRuntime | TextRuntime | None = None
    last_used_at: float | None = None


class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._cleanup_thread: threading.Thread | None = None
        self._entries = {alias: ModelEntry(spec=spec) for alias, spec in MODEL_SPECS.items()}

    def start(self) -> None:
        if self._cleanup_thread is not None:
            return

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="model-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()
        logger.info(
            "Model cleanup thread started: idle_unload=%ss interval=%ss",
            MODEL_IDLE_UNLOAD_SECONDS,
            MODEL_CLEANUP_INTERVAL_SECONDS,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=2)
            self._cleanup_thread = None

        with self._lock:
            for entry in self._entries.values():
                if entry.runtime is not None:
                    entry.runtime.unload()
                    entry.runtime = None
                    entry.last_used_at = None

    def _cleanup_loop(self) -> None:
        while not self._stop_event.wait(MODEL_CLEANUP_INTERVAL_SECONDS):
            self.unload_idle_models()

    def unload_idle_models(self) -> None:
        now = time.time()
        with self._lock:
            for entry in self._entries.values():
                if entry.runtime is None or entry.last_used_at is None:
                    continue
                idle_seconds = now - entry.last_used_at
                if idle_seconds >= MODEL_IDLE_UNLOAD_SECONDS:
                    entry.runtime.unload()
                    entry.runtime = None
                    entry.last_used_at = None
                    logger.info(
                        "Idle unload triggered: alias=%s idle=%ss",
                        entry.spec.alias,
                        int(idle_seconds),
                    )

    def _resolve(self, selector: str | None) -> ModelEntry:
        if selector is None or selector.strip() == "":
            return self._entries[DEFAULT_MODEL_ALIAS]

        normalized = selector.strip()
        if normalized in {"qwen-0.5b", "gemma-3-1b"}:
            normalized = "qwen-1.5b"
        if normalized in self._entries:
            return self._entries[normalized]

        for entry in self._entries.values():
            if entry.spec.model_id == normalized:
                return entry

        available = [f"{e.spec.alias} ({e.spec.model_id})" for e in self._entries.values()]
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{selector}'. Available: {', '.join(available)}",
        )

    def _ensure_runtime_loaded(self, entry: ModelEntry) -> VisionRuntime | TextRuntime:
        runtime = entry.runtime
        if runtime is None:
            if entry.spec.kind == "vision":
                runtime = VisionRuntime(entry.spec)
            else:
                runtime = TextRuntime(entry.spec)
            entry.runtime = runtime

        if runtime.is_loaded():
            return runtime

        try:
            runtime.load()
            return runtime
        except Exception as exc:
            logger.exception(
                "Failed to load model: alias=%s model=%s",
                entry.spec.alias,
                entry.spec.model_id,
            )
            try:
                runtime.unload()
            except Exception:
                logger.exception("Failed to unload model after load error: alias=%s", entry.spec.alias)
            entry.runtime = None
            entry.last_used_at = None
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load model '{entry.spec.alias}': {exc}",
            ) from exc

    def acquire(
        self,
        model_selector: str | None,
        requires_vision: bool,
    ) -> tuple[ModelSpec, VisionRuntime | TextRuntime]:
        with self._lock:
            entry = self._resolve(model_selector)
            if requires_vision and entry.spec.kind != "vision":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Model '{entry.spec.alias}' does not support image input. "
                        "Use 'smolvlm-256m' for image requests."
                    ),
                )

            if SINGLE_ACTIVE_MODEL:
                for alias, other in self._entries.items():
                    if alias == entry.spec.alias:
                        continue
                    if other.runtime is not None:
                        other.runtime.unload()
                        other.runtime = None
                        other.last_used_at = None

            runtime = self._ensure_runtime_loaded(entry)
            entry.last_used_at = time.time()
            return entry.spec, runtime

    def status(self) -> list[dict[str, Any]]:
        now = time.time()
        with self._lock:
            result: list[dict[str, Any]] = []
            for entry in self._entries.values():
                idle_seconds = None
                if entry.last_used_at is not None:
                    idle_seconds = int(max(0, now - entry.last_used_at))
                result.append(
                    {
                        "alias": entry.spec.alias,
                        "model_id": entry.spec.model_id,
                        "kind": entry.spec.kind,
                        "loaded": entry.runtime is not None,
                        "idle_seconds": idle_seconds,
                    }
                )
            return result


model_registry = ModelRegistry()


def _open_uploaded_image(raw_bytes: bytes) -> Image.Image:
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Image is empty")

    try:
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image") from exc


def _open_image_from_url(url: str) -> Image.Image:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="image_url must use http or https")

    try:
        with urlopen(url, timeout=30) as response:
            raw = response.read()
    except URLError as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image_url: {exc}") from exc

    return _open_uploaded_image(raw)


def _open_image_from_data_url(data_url: str) -> Image.Image:
    if "," not in data_url:
        raise HTTPException(status_code=400, detail="Invalid data URL")

    header, payload = data_url.split(",", 1)
    if ";base64" not in header:
        raise HTTPException(status_code=400, detail="Only base64 data URLs are supported")

    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc

    return _open_uploaded_image(raw)


def _normalize_chat_messages(messages: list[ChatMessage]) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    hf_messages: list[dict[str, Any]] = []
    images: list[Image.Image] = []

    for message in messages:
        role = message.role.strip().lower()
        if role not in {"system", "user", "assistant"}:
            raise HTTPException(
                status_code=400,
                detail="message.role must be one of: system, user, assistant",
            )

        if isinstance(message.content, str):
            text = message.content.strip()
            if not text:
                raise HTTPException(status_code=400, detail="message.content cannot be empty")
            hf_messages.append({"role": role, "content": [{"type": "text", "text": text}]})
            continue

        content_items: list[dict[str, str]] = []
        for item in message.content:
            if item.type == "text":
                text = (item.text or "").strip()
                if not text:
                    raise HTTPException(status_code=400, detail="text content cannot be empty")
                content_items.append({"type": "text", "text": text})
                continue

            if item.type == "image_url":
                if item.image_url is None or not item.image_url.url:
                    raise HTTPException(status_code=400, detail="image_url.url is required")

                image_url = item.image_url.url.strip()
                if image_url.startswith("data:"):
                    image = _open_image_from_data_url(image_url)
                else:
                    image = _open_image_from_url(image_url)

                images.append(image)
                content_items.append({"type": "image"})
                continue

            raise HTTPException(status_code=400, detail=f"Unsupported content type: {item.type}")

        if not content_items:
            raise HTTPException(status_code=400, detail="message.content cannot be empty")
        hf_messages.append({"role": role, "content": content_items})

    return hf_messages, images


def _validate_max_new_tokens(value: int) -> int:
    if value < 1 or value > MAX_NEW_TOKENS_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS_LIMIT}",
        )
    return value


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting model API")
    model_registry.start()
    yield
    logger.info("Stopping model API")
    model_registry.stop()


app = FastAPI(
    title="Model API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "device": DEVICE,
        "default_model_alias": DEFAULT_MODEL_ALIAS,
        "single_active_model": SINGLE_ACTIVE_MODEL,
        "model_idle_unload_seconds": MODEL_IDLE_UNLOAD_SECONDS,
        "model_cleanup_interval_seconds": MODEL_CLEANUP_INTERVAL_SECONDS,
        "models": model_registry.status(),
    }


@app.get("/models")
def models() -> dict[str, Any]:
    return {
        "default_model_alias": DEFAULT_MODEL_ALIAS,
        "available": model_registry.status(),
    }


@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
    model: str | None = Form(None),
) -> dict[str, Any]:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    clean_prompt = prompt.strip()
    if not clean_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    raw = await image.read()
    pil_image = _open_uploaded_image(raw)
    safe_max_new_tokens = _validate_max_new_tokens(max_new_tokens)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": clean_prompt},
            ],
        }
    ]

    try:
        spec, runtime = model_registry.acquire(model_selector=model, requires_vision=True)
        generated_text = runtime.generate_chat(
            messages=messages,
            images=[pil_image],
            max_new_tokens=safe_max_new_tokens,
        )
    except RuntimeError as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "model_alias": spec.alias,
        "model_id": spec.model_id,
        "prompt": clean_prompt,
        "generated_text": generated_text,
        "max_new_tokens": safe_max_new_tokens,
    }


@app.post("/chat")
async def chat(request: ChatCompletionRequest) -> dict[str, Any]:
    raw_tokens = request.max_new_tokens
    if raw_tokens is None:
        raw_tokens = request.max_tokens if request.max_tokens is not None else DEFAULT_MAX_NEW_TOKENS
    safe_max_new_tokens = _validate_max_new_tokens(raw_tokens)

    hf_messages, images = _normalize_chat_messages(request.messages)

    try:
        spec, runtime = model_registry.acquire(
            model_selector=request.model,
            requires_vision=len(images) > 0,
        )
        generated_text = runtime.generate_chat(
            messages=hf_messages,
            images=images,
            max_new_tokens=safe_max_new_tokens,
        )
    except RuntimeError as exc:
        logger.exception("Chat generation failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "model_alias": spec.alias,
        "model_id": spec.model_id,
        "assistant_message": generated_text,
        "max_new_tokens": safe_max_new_tokens,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    raw_tokens = request.max_new_tokens
    if raw_tokens is None:
        raw_tokens = request.max_tokens if request.max_tokens is not None else DEFAULT_MAX_NEW_TOKENS
    safe_max_new_tokens = _validate_max_new_tokens(raw_tokens)

    hf_messages, images = _normalize_chat_messages(request.messages)

    try:
        spec, runtime = model_registry.acquire(
            model_selector=request.model,
            requires_vision=len(images) > 0,
        )
        generated_text = runtime.generate_chat(
            messages=hf_messages,
            images=images,
            max_new_tokens=safe_max_new_tokens,
        )
    except RuntimeError as exc:
        logger.exception("Chat completion failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": spec.model_id,
        "model_alias": spec.alias,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": "stop",
            }
        ],
    }
