"""Aturan masukan. Dipakai API dan Streamlit Mode A, jadi input yang ditolak di keduanya sama persis.

Tiga lapis penolakan:
1. kosong / bukan angka / NaN / kolom asing
2. mustahil secara fisik (PM2.5 negatif, kelembapan > 100%, arah angin > 360°, ...)
3. di luar rentang data latih (mungkin secara fisik, tapi model belum pernah melihatnya -> tidak bisa dipercaya)
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app import predictor
from src.features import STASIUN

CONTOH = {
    "stasiun": "Dago", "waktu": "2026-06-15T07:00:00",
    "pm25_ugm3": 35, "pm25_lag1": 33, "pm25_lag3": 30, "pm25_lag24": 28,
    "suhu_C": 22, "kelembaban_pct": 80, "kecepatan_angin_ms": 1.5,
    "arah_angin_deg": 180, "curah_hujan_mm": 0,
}

# kolom masukan -> kolom acuan rentangnya di metadata
ACUAN_RENTANG = {
    "pm25_ugm3": "pm25_ugm3", "pm25_lag1": "pm25_ugm3", "pm25_lag3": "pm25_ugm3", "pm25_lag24": "pm25_ugm3",
    "suhu_C": "suhu_C", "kelembaban_pct": "kelembaban_pct", "kecepatan_angin_ms": "kecepatan_angin_ms",
    "arah_angin_deg": "arah_angin_deg", "curah_hujan_mm": "curah_hujan_mm",
}


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, json_schema_extra={"examples": [CONTOH]})

    stasiun: Literal[tuple(STASIUN)]
    waktu: datetime = Field(description="Tanggal & jam pengukuran terakhir (WIB), contoh 2026-06-15T07:00")
    pm25_ugm3: float = Field(ge=0, le=1000, description="PM2.5 sekarang (µg/m³)")
    pm25_lag1: float = Field(ge=0, le=1000, description="PM2.5 1 jam lalu")
    pm25_lag3: float = Field(ge=0, le=1000, description="PM2.5 3 jam lalu")
    pm25_lag24: float = Field(ge=0, le=1000, description="PM2.5 24 jam lalu")
    suhu_C: float = Field(ge=-10, le=50)
    kelembaban_pct: float = Field(ge=0, le=100)
    kecepatan_angin_ms: float = Field(ge=0, le=75)
    arah_angin_deg: float = Field(ge=0, le=360)
    curah_hujan_mm: float = Field(ge=0, le=500)

    @model_validator(mode="after")
    def dalam_rentang_data_latih(self):
        rentang = predictor.load()[1]["rentang_input_data_latih"]
        salah = []
        for kolom, acuan in ACUAN_RENTANG.items():
            nilai, r = getattr(self, kolom), rentang[acuan]
            if not r["min"] <= nilai <= r["max"]:
                salah.append(f"{kolom}={nilai:g} di luar rentang data latih ({r['min']:g} – {r['max']:g})")
        if salah:
            raise ValueError("; ".join(salah))
        return self


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


def terjemahkan_error(errors) -> list[str]:
    """Ubah error Pydantic jadi pesan bahasa Indonesia: 'kolom: alasan'."""
    pesan = []
    for e in errors:
        kolom = ".".join(str(x) for x in e["loc"] if x != "body")
        tipe, ctx = e["type"], e.get("ctx", {})
        if tipe == "greater_than_equal":
            teks = f"minimal {ctx['ge']}"
        elif tipe == "less_than_equal":
            teks = f"maksimal {ctx['le']}"
        elif tipe == "literal_error":
            teks = f"harus salah satu dari: {ctx['expected']}"
        elif tipe == "value_error":
            teks = e["msg"].removeprefix("Value error, ")
        else:
            teks = TERJEMAHAN.get(tipe, e["msg"])
        pesan.append(f"{kolom}: {teks}" if kolom else teks)
    return pesan
