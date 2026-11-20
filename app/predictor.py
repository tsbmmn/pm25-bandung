"""Muat artifact sekali, lalu prediksi.
Dipakai FastAPI (app/api/main.py) dan Streamlit Mode A (app/ui/streamlit_app.py),
jadi hasil prediksi di API dan di aplikasi dijamin sama.
"""
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # supaya `src` bisa di-import dari mana pun file ini dijalankan
from src.features import tambah_fitur  # noqa: E402  (fungsi fitur yang SAMA dengan saat pelatihan)

ARTIFACT_DIR = ROOT / "artifacts"


@lru_cache(maxsize=1)
def load():
    """(model, metadata). lru_cache = di-load sekali per proses, bukan setiap prediksi."""
    model_path, meta_path = ARTIFACT_DIR / "model.joblib", ARTIFACT_DIR / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} tidak ada. Sudah di-commit? Cek .gitignore.")
    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, meta


def kategori_dan_saran(pm25: float, ambang_alarm: float) -> tuple[str, str]:
    """Label tampilan saja (acuan kategori PM2.5 ISPU). Tidak memengaruhi model."""
    if pm25 > 55.4:
        return "Tidak sehat", ("Kurangi aktivitas di luar ruangan dan pakai masker. Penderita asma, lansia, "
                               "anak-anak, dan ibu hamil sebaiknya tetap di dalam ruangan.")
    if pm25 > ambang_alarm:
        return "Sedang", ("Udara mendekati tidak sehat. Penderita asma dan kelompok sensitif sebaiknya mulai "
                          "memakai masker dan membatasi aktivitas di luar ruangan.")
    if pm25 > 15.5:
        return "Sedang", ("Aman untuk aktivitas biasa. Kelompok sensitif sebaiknya tidak berolahraga berat "
                          "di luar ruangan terlalu lama.")
    return "Baik", "Udara bersih. Aman untuk beraktivitas di luar ruangan."


def predict_rows(rows: list[dict]) -> list[dict]:
    """Prediksi beberapa baris sekaligus. Tiap baris berisi kolom masukan (stasiun, waktu, PM2.5, cuaca)."""
    model, meta = load()
    mulai = time.perf_counter()
    df = pd.DataFrame(rows)
    pred = model.predict(tambah_fitur(df))
    latency = round((time.perf_counter() - mulai) * 1000, 1)

    hasil = []
    for waktu, stasiun, y in zip(pd.to_datetime(df["waktu"]), df["stasiun"], pred):
        y = float(y)
        kategori, saran = kategori_dan_saran(y, meta["ambang_alarm"])
        hasil.append({
            "stasiun": stasiun,
            "waktu_prediksi": (waktu + pd.Timedelta(hours=meta["horizon_jam"])).isoformat(),
            "prediksi_pm25_ugm3": round(y, 1),
            "kategori": kategori,
            "peringatan": y > meta["ambang_alarm"],
            "saran": saran,
            "ambang_alarm": meta["ambang_alarm"],
            "versi_model": meta["versi_model"],
            "latency_ms": latency,
        })
    return hasil


def predict_one(row: dict) -> dict:
    return predict_rows([row])[0]
