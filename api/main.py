"""API prediksi PM2.5 3 jam ke depan.
Jalankan dari folder repo:  uvicorn api.main:app --reload
"""
import json, sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.features import STASIUN, tambah_fitur   # fungsi yang SAMA dengan pelatihan

MODEL = joblib.load(ROOT / "models" / "model.joblib")
META = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
RENTANG = META["rentang_input_data_latih"]
AMBANG = META["ambang_alarm"]

app = FastAPI(title="API Prediksi PM2.5 Bandung", version=META["versi_model"])


class InputPrediksi(BaseModel):
    # tolak kolom asing dan nilai NaN/inf
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    stasiun: str
    waktu: datetime = Field(description="Tanggal & jam pengukuran terakhir, contoh 2026-06-15T07:00")
    pm25_ugm3: float = Field(ge=0, le=1000, description="PM2.5 sekarang")
    pm25_lag1: float = Field(ge=0, le=1000, description="PM2.5 1 jam lalu")
    pm25_lag3: float = Field(ge=0, le=1000, description="PM2.5 3 jam lalu")
    pm25_lag24: float = Field(ge=0, le=1000, description="PM2.5 24 jam lalu")
    suhu_C: float = Field(ge=-10, le=50)
    kelembaban_pct: float = Field(ge=0, le=100)
    kecepatan_angin_ms: float = Field(ge=0, le=75)
    arah_angin_deg: float = Field(ge=0, le=360)
    curah_hujan_mm: float = Field(ge=0, le=500)

    @model_validator(mode="after")
    def cek_wajar(self):
        salah = []
        if self.stasiun not in STASIUN:
            salah.append(f"stasiun harus salah satu dari: {', '.join(STASIUN)}")
        # Nilai yang mungkin secara fisik tapi di luar data latih juga ditolak,
        # karena model tidak bisa dipercaya di luar rentang yang pernah dilihatnya.
        cek = {
            "pm25_ugm3": "pm25_ugm3", "pm25_lag1": "pm25_ugm3",
            "pm25_lag3": "pm25_ugm3", "pm25_lag24": "pm25_ugm3",
            "suhu_C": "suhu_C", "kelembaban_pct": "kelembaban_pct",
            "kecepatan_angin_ms": "kecepatan_angin_ms",
            "arah_angin_deg": "arah_angin_deg", "curah_hujan_mm": "curah_hujan_mm",
        }
        for kolom, acuan in cek.items():
            nilai, r = getattr(self, kolom), RENTANG[acuan]
            if not r["min"] <= nilai <= r["max"]:
                salah.append(f"{kolom}={nilai} di luar rentang data latih ({r['min']:g} – {r['max']:g})")
        if salah:
            raise ValueError("; ".join(salah))
        return self


# Pesan error dalam bahasa Indonesia (penting buat demo video)
TERJEMAHAN = {
    "missing": "wajib diisi",
    "float_parsing": "harus berupa angka",
    "float_type": "harus berupa angka",
    "finite_number": "harus angka biasa (bukan NaN/inf)",
    "extra_forbidden": "kolom tidak dikenal",
    "datetime_parsing": "format waktu salah (contoh: 2026-06-15T07:00)",
    "datetime_from_date_parsing": "format waktu salah (contoh: 2026-06-15T07:00)",
    "datetime_type": "format waktu salah (contoh: 2026-06-15T07:00)",
}

@app.exception_handler(RequestValidationError)
async def tolak_input(request: Request, exc: RequestValidationError):
    pesan = []
    for e in exc.errors():
        kolom = ".".join(str(x) for x in e["loc"] if x != "body")
        tipe, ctx = e["type"], e.get("ctx", {})
        if tipe == "greater_than_equal":
            teks = f"minimal {ctx['ge']}"
        elif tipe == "less_than_equal":
            teks = f"maksimal {ctx['le']}"
        elif tipe == "value_error":
            teks = e["msg"].removeprefix("Value error, ")
        else:
            teks = TERJEMAHAN.get(tipe, e["msg"])
        pesan.append(f"{kolom}: {teks}" if kolom else teks)
    return JSONResponse(status_code=422, content={"ditolak": True, "pesan": pesan})


def kategori(x: float) -> str:
    # Label tampilan saja (acuan ISPU PM2.5). Konfirmasi batasnya dengan Denia.
    if x <= 15.5:
        return "Baik"
    if x <= 55.4:
        return "Sedang"
    return "Tidak sehat"


@app.get("/health")
def health():
    return {"status": "ok", "versi_model": META["versi_model"]}


@app.get("/info")
def info():
    return {k: META[k] for k in ["versi_model", "target", "stasiun", "ambang_alarm",
                                 "metrik_test", "rentang_input_data_latih"]}


@app.post("/predict")
def predict(inp: InputPrediksi):
    X = tambah_fitur(pd.DataFrame([inp.model_dump()]))
    y = float(MODEL.predict(X)[0])
    return {
        "stasiun": inp.stasiun,
        "waktu_input": inp.waktu,
        "waktu_prediksi": inp.waktu + timedelta(hours=META["horizon_jam"]),
        "prediksi_pm25_ugm3": round(y, 1),
        "kategori": kategori(y),
        "peringatan": y > AMBANG,
        "ambang_alarm": AMBANG,
        "versi_model": META["versi_model"],
    }