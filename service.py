"""Serve offline image QA through a local vLLM vision endpoint."""
import base64
import io
import os
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

app = FastAPI(title="Driving Scene VLM")
MAX_BYTES = 5 * 1024 * 1024
BACKEND = os.getenv("VLLM_URL", "http://localhost:8001/v1").rstrip("/")
MODEL = os.getenv("MODEL", "driving-scene")


class Prediction(BaseModel):
    answer: str
    model: str


@app.post("/predict", response_model=Prediction)
async def predict(
    image: Annotated[UploadFile, File()],
    question: Annotated[str, Form(min_length=1, max_length=2000)],
) -> Prediction:
    raw = await image.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Image must be at most 5 MiB")
    if not question.strip():
        raise HTTPException(422, "Question must not be blank")
    try:
        with Image.open(io.BytesIO(raw)) as frame:
            if frame.format not in {"JPEG", "PNG"} or frame.width * frame.height > 16_000_000:
                raise ValueError("Unsupported frame")
            mime = Image.MIME[frame.format]
            frame.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError):
        raise HTTPException(422, "Expected a valid JPEG or PNG frame of at most 16 megapixels") from None
    uri = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    payload = {
        "model": MODEL, "max_tokens": 96, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": question.strip()},
        ]}],
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{BACKEND}/chat/completions", json=payload)
            response.raise_for_status()
        return Prediction(answer=response.json()["choices"][0]["message"]["content"], model=MODEL)
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        raise HTTPException(502, "The vision backend did not return a usable answer") from None
