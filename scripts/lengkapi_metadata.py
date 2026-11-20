"""Lengkapi artifacts/metadata.json TANPA mengubah model:
- fitur_input: kolom yang diisi pengguna di API/aplikasi (checklist README_TIM)
- metrik_baseline: baseline wajib "persistence" (3 jam lagi = sekarang) di validation dan test

Jalankan sekali dari root repo:  python scripts/lengkapi_metadata.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import buat_lag  # noqa: E402

META_PATH = ROOT / "artifacts" / "metadata.json"
meta = json.loads(META_PATH.read_text(encoding="utf-8"))

# Persiapan data sama persis dengan notebook dan train_final.py
df = pd.read_csv(ROOT / "data" / "clean.csv", parse_dates=["waktu"])
df["target"] = df.groupby("stasiun")["pm25_ugm3"].shift(-meta["horizon_jam"])
d = buat_lag(df).dropna(subset=["pm25_ugm3", "target"])
w = d["waktu"]


def evaluasi_persistence(mask):
    pred = d.loc[mask, "pm25_ugm3"].to_numpy()       # baseline: 3 jam lagi = sekarang
    y = d.loc[mask, "target"].to_numpy()
    alarm, kejadian = pred > meta["ambang_alarm"], y > meta["ambang_kejadian"]
    return {"mae": round(float(mean_absolute_error(y, pred)), 3),
            "recall": round(float(recall_score(kejadian, alarm, zero_division=0)), 3),
            "precision": round(float(precision_score(kejadian, alarm, zero_division=0)), 3)}


val = (w >= "2026-03-01") & (w < "2026-05-01")
test = (w >= "2026-05-01") & (w < "2026-07-01")
meta["metrik_baseline"] = {"nama": "persistence (PM2.5 3 jam lagi = PM2.5 sekarang)",
                           "validation": evaluasi_persistence(val), "test": evaluasi_persistence(test)}
meta["fitur_input"] = ["stasiun", "waktu", "pm25_ugm3", "pm25_lag1", "pm25_lag3", "pm25_lag24",
                       "suhu_C", "kelembaban_pct", "kecepatan_angin_ms", "arah_angin_deg", "curah_hujan_mm"]

META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print("Baseline validation:", meta["metrik_baseline"]["validation"])
print("Baseline test      :", meta["metrik_baseline"]["test"])
print("Model test         :", {k: meta["metrik_test"][k] for k in ("mae", "recall", "precision")})
