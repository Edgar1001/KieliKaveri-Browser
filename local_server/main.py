import asyncio
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from faster_whisper import WhisperModel
from openai import APIError, AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from starlette.background import BackgroundTask

from local_server.tutor import (
    HistoryTurn,
    LanguageMode,
    Level,
    Topic,
    TutorResponse,
    build_transcription_prompt,
    build_tutor_prompt,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("language-tutor")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8_float16")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_ADMIN_API_KEY = os.getenv("OPENAI_ADMIN_API_KEY", "")
OPENAI_USAGE_PROJECT_ID = os.getenv("OPENAI_USAGE_PROJECT_ID", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
OPENAI_BUDGET_USD = float(os.getenv("OPENAI_BUDGET_USD", "5"))
OPENAI_INPUT_COST_PER_MILLION = float(os.getenv("OPENAI_INPUT_COST_PER_MILLION", "2"))
OPENAI_CACHED_INPUT_COST_PER_MILLION = float(
    os.getenv("OPENAI_CACHED_INPUT_COST_PER_MILLION", "0.5")
)
OPENAI_OUTPUT_COST_PER_MILLION = float(os.getenv("OPENAI_OUTPUT_COST_PER_MILLION", "8"))
USAGE_STATE_PATH = Path(
    os.getenv("USAGE_STATE_PATH", Path(__file__).parent.parent / "server-data" / "usage.json")
)
PIPER_EXECUTABLE = os.getenv("PIPER_EXECUTABLE", str(Path(sys.executable).with_name("piper")))
PIPER_VOICE = Path(
    os.getenv(
        "PIPER_VOICE",
        Path(__file__).parent / "voices" / "fi_FI-harri-medium.onnx",
    )
)

HISTORY_ADAPTER = TypeAdapter(list[HistoryTurn])
TOPIC_ADAPTER = TypeAdapter(Topic)
LEVEL_ADAPTER = TypeAdapter(Level)
LANGUAGE_MODE_ADAPTER = TypeAdapter(LanguageMode)
MODEL_LOCK = threading.Lock()
INFERENCE_LOCK = asyncio.Lock()
usage_lock = asyncio.Lock()
whisper_runtime = WHISPER_DEVICE

app = FastAPI(title="Kielikaveri API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=600)


def load_estimated_spend() -> float:
    try:
        data = json.loads(USAGE_STATE_PATH.read_text(encoding="utf-8"))
        return max(0.0, float(data["estimatedSpendUsd"]))
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0.0


def save_estimated_spend(spend: float) -> None:
    USAGE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = USAGE_STATE_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps({"estimatedSpendUsd": spend}), encoding="utf-8")
    temporary_path.replace(USAGE_STATE_PATH)


estimated_spend_usd = load_estimated_spend()


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Any, error: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"error": str(error.detail)})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Any, _error: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": "Invalid request data."})


def transcribe_with_model(audio_path: Path, initial_prompt: str) -> str:
    LOGGER.info(
        "Loading Whisper model %s on %s (%s)",
        WHISPER_MODEL,
        whisper_runtime,
        WHISPER_COMPUTE_TYPE if whisper_runtime == "cuda" else "int8",
    )
    model = WhisperModel(
        WHISPER_MODEL,
        device=whisper_runtime,
        compute_type=WHISPER_COMPUTE_TYPE if whisper_runtime == "cuda" else "int8",
    )
    try:
        segments, _ = model.transcribe(
            str(audio_path),
            language="fi",
            beam_size=5,
            vad_filter=True,
            vad_parameters={
                "threshold": 0.65,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
            condition_on_previous_text=False,
            hallucination_silence_threshold=1.0,
            initial_prompt=initial_prompt,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        del model
        gc.collect()


def transcribe_audio(audio_path: Path, initial_prompt: str) -> str:
    global whisper_runtime

    with MODEL_LOCK:
        try:
            return transcribe_with_model(audio_path, initial_prompt)
        except Exception:
            if whisper_runtime != "cuda":
                raise
            LOGGER.exception("CUDA transcription failed; retrying on CPU")
            whisper_runtime = "cpu"
            return transcribe_with_model(audio_path, initial_prompt)


def normalize_speech_audio(audio_path: Path) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as normalized_file:
        normalized_path = Path(normalized_file.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio_path),
                "-af",
                "highpass=f=80,lowpass=f=7500,dynaudnorm=f=150:g=15:p=0.95",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(normalized_path),
            ],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        normalized_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The recorded audio could not be processed.") from error
    return normalized_path


async def generate_tutor_response(
    transcript: str,
    topic: Topic,
    history: list[HistoryTurn],
    level: Level = "B2",
    language_mode: LanguageMode = "Puhekieli",
) -> TutorResponse:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")

    try:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=60)
        response = await client.responses.parse(
            model=OPENAI_MODEL,
            instructions=build_tutor_prompt(topic, history, level, language_mode),
            input=f"Finnish learner utterance: {transcript}\nWrite the explanation in English only.",
            text_format=TutorResponse,
            temperature=0.1,
            max_output_tokens=500,
        )
    except APIError as error:
        LOGGER.exception("OpenAI tutor request failed")
        raise HTTPException(status_code=503, detail="The OpenAI tutor is unavailable.") from error

    if response.output_parsed is None:
        raise HTTPException(status_code=502, detail="The OpenAI tutor returned invalid output.")
    usage = getattr(response, "usage", None)
    if usage is not None:
        input_tokens = usage.input_tokens or 0
        input_details = getattr(usage, "input_tokens_details", None)
        cached_input_tokens = getattr(input_details, "cached_tokens", 0) or 0
        uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
        request_cost = (
            uncached_input_tokens * OPENAI_INPUT_COST_PER_MILLION
            + cached_input_tokens * OPENAI_CACHED_INPUT_COST_PER_MILLION
            + (usage.output_tokens or 0) * OPENAI_OUTPUT_COST_PER_MILLION
        ) / 1_000_000
        async with usage_lock:
            global estimated_spend_usd
            estimated_spend_usd += request_cost
            save_estimated_spend(estimated_spend_usd)
    return response.output_parsed


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok" if OPENAI_API_KEY else "setup_required",
        "openaiConfigured": bool(OPENAI_API_KEY),
        "openaiModel": OPENAI_MODEL,
        "whisperModel": WHISPER_MODEL,
        "whisperDevice": whisper_runtime,
    }


@app.get("/api/usage")
async def usage() -> dict[str, float | str]:
    if OPENAI_ADMIN_API_KEY:
        month_start = datetime.now(timezone.utc).replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        query: dict[str, Any] = {
            "start_time": int(month_start.timestamp()),
            "bucket_width": "1d",
            "limit": 31,
        }
        if OPENAI_USAGE_PROJECT_ID:
            query["project_ids"] = [OPENAI_USAGE_PROJECT_ID]
        try:
            client = AsyncOpenAI(admin_api_key=OPENAI_ADMIN_API_KEY, timeout=15)
            costs = await client.admin.organization.usage.costs(**query)
            spend = sum(
                result.amount.value
                for bucket in costs.data
                for result in bucket.results
                if result.object == "organization.costs.result"
                and result.amount is not None
                and result.amount.value is not None
                and result.amount.currency == "usd"
            )
            percentage = min(100.0, spend / OPENAI_BUDGET_USD * 100) if OPENAI_BUDGET_USD else 0.0
            return {
                "estimatedSpendUsd": round(spend, 6),
                "budgetUsd": OPENAI_BUDGET_USD,
                "percentage": round(percentage, 2),
                "scope": "OpenAI Platform, tämä kuukausi",
            }
        except OpenAIError:
            LOGGER.warning("OpenAI platform costs unavailable; using local usage", exc_info=True)

    async with usage_lock:
        spend = estimated_spend_usd
    percentage = min(100.0, spend / OPENAI_BUDGET_USD * 100) if OPENAI_BUDGET_USD else 0.0
    return {
        "estimatedSpendUsd": round(spend, 6),
        "budgetUsd": OPENAI_BUDGET_USD,
        "percentage": round(percentage, 2),
        "scope": "Sovelluksen paikallinen seuranta",
    }


@app.post("/api/speech", response_class=FileResponse)
async def speech(request: SpeechRequest) -> FileResponse:
    if not Path(PIPER_EXECUTABLE).is_file() or not PIPER_VOICE.is_file():
        raise HTTPException(status_code=503, detail="Finnish neural speech is not installed.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
        output_path = Path(output_file.name)
    try:
        await asyncio.to_thread(
            subprocess.run,
            [
                PIPER_EXECUTABLE,
                "--model",
                str(PIPER_VOICE),
                "--output_file",
                str(output_path),
            ],
            input=request.text,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        output_path.unlink(missing_ok=True)
        LOGGER.error("Piper synthesis failed: %s", error.stderr.strip())
        raise HTTPException(status_code=503, detail="Finnish speech synthesis failed.") from error

    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="kielikaveri.wav",
        background=BackgroundTask(output_path.unlink, missing_ok=True),
    )


@app.post("/api/conversation")
async def conversation(
    audio: UploadFile = File(...),
    topic: str = Form("Arki"),
    level: str = Form("B2"),
    languageMode: str = Form("Puhekieli"),
    history: str = Form("[]"),
) -> dict[str, str]:
    try:
        parsed_topic = TOPIC_ADAPTER.validate_python(topic)
        parsed_level = LEVEL_ADAPTER.validate_python(level)
        parsed_language_mode = LANGUAGE_MODE_ADAPTER.validate_python(languageMode)
        parsed_history = HISTORY_ADAPTER.validate_json(history)
        if len(parsed_history) > 4:
            raise ValueError("Conversation history is too long.")
    except (ValidationError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Invalid conversation data.") from error

    suffix = Path(audio.filename or "speech.m4a").suffix or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
        shutil.copyfileobj(audio.file, temporary_file)

    normalized_path: Path | None = None
    try:
        async with INFERENCE_LOCK:
            normalized_path = await asyncio.to_thread(normalize_speech_audio, temporary_path)
            transcription_prompt = build_transcription_prompt(
                parsed_topic,
                parsed_history,
                parsed_language_mode,
            )
            transcript = await asyncio.to_thread(
                transcribe_audio,
                normalized_path,
                transcription_prompt,
            )
            if not transcript:
                raise HTTPException(status_code=422, detail="No Finnish speech was detected. Please try again.")
            tutor_response = await generate_tutor_response(
                transcript,
                parsed_topic,
                parsed_history,
                parsed_level,
                parsed_language_mode,
            )
        return {"transcript": transcript, **tutor_response.model_dump()}
    finally:
        temporary_path.unlink(missing_ok=True)
        if normalized_path is not None:
            normalized_path.unlink(missing_ok=True)