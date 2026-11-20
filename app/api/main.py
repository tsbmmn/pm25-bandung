"""FastAPI untuk prakiraan PM2.5 Bandung 3 jam ke depan.

Jalankan dari root repo:
    uvicorn app.api.main:app --reload
Lalu buka http://127.0.0.1:8000/docs
"""
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # supaya `app` dan `src` bisa di-import
from app import predictor  # noqa: E402
from app.schema import InputData, terjemahkan_error  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prediksi")

MAX_BATCH = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()  # 1 · load model sekali saat startup, bukan per request
    yield


app = FastAPI(title="Prakiraan PM2.5 Bandung", version="1.0.0", lifespan=lifespan)


class Prediksi(BaseModel):
    stasiun: str
    waktu_prediksi: str
    prediksi_pm25_ugm3: float
    kategori: str
    peringatan: bool
    saran: str
    ambang_alarm: float
    versi_model: str
    latency_ms: float


@app.exception_handler(RequestValidationError)
async def tolak_input(request: Request, exc: RequestValidationError):
    """2 · Input yang tidak wajar ditolak 422 dengan pesan bahasa Indonesia, sebelum menyentuh model."""
    return JSONResponse(status_code=422, content={"ditolak": True, "pesan": terjemahkan_error(exc.errors())})


def _log(inputs: list[dict], outputs: list[dict]):
    """Setiap prediksi di-log (input + output) supaya bisa diaudit dan dipantau perubahan polanya."""
    for x, y in zip(inputs, outputs):
        log.info("prediksi %s", json.dumps({"input": x, "output": y, "ts": time.time()}, default=str))


@app.get("/health")
def health():
    _, meta = predictor.load()
    return {"status": "ok", "versi_model": meta["versi_model"], "target": meta["target"],
            "horizon_jam": meta["horizon_jam"], "ambang_alarm": meta["ambang_alarm"],
            "stasiun": meta["stasiun"], "metrik_test": meta["metrik_test"],
            "rentang_input_data_latih": meta["rentang_input_data_latih"]}


@app.post("/predict", response_model=Prediksi)
def predict(x: InputData):
    """3 · JSON masuk, prakiraan PM2.5 3 jam lagi keluar."""
    row = x.model_dump()
    hasil = predictor.predict_one(row)
    _log([row], [hasil])
    return hasil


@app.post("/predict/batch", response_model=list[Prediksi])
def predict_batch(xs: list[InputData]):
    """Beberapa stasiun / jam sekaligus (maks 1000)."""
    if len(xs) > MAX_BATCH:
        raise HTTPException(413, f"Maksimal {MAX_BATCH} baris per request.")
    rows = [x.model_dump() for x in xs]
    if not rows:
        return []
    hasil = predictor.predict_rows(rows)
    _log(rows, hasil)
    return hasil
