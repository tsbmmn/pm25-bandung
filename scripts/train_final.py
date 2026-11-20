"""Latih model final + evaluasi test SATU KALI.
Jalankan dari folder repo:  python scripts/train_final.py
"""
import json, sys, platform
from datetime import datetime
from pathlib import Path

import joblib, numpy as np, pandas as pd, sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.features import FITUR, STASIUN, buat_lag, tambah_fitur

AMBANG_ALARM = 45      # alarm kalau prediksi > 45
AMBANG_KEJADIAN = 55   # "kejadian" = PM2.5 aktual > 55
HORIZON_JAM = 3        # target = PM2.5 3 jam ke depan
AWAL_TEST = pd.Timestamp("2026-05-01")
AKHIR_TEST = pd.Timestamp("2026-07-01")
META_PATH = ROOT / "artifacts" / "metadata.json"

# Pengaman: test hanya boleh dibuka sekali
if META_PATH.exists() and "--paksa" not in sys.argv:
    sys.exit("metadata.json sudah ada -> test sudah pernah dibuka. Jangan dijalankan ulang.")

# --- Data & fitur (sama persis dengan notebook Denia) ---
df = pd.read_csv(ROOT / "data" / "clean.csv", parse_dates=["waktu"])
df["target"] = df.groupby("stasiun")["pm25_ugm3"].shift(-HORIZON_JAM)
d = buat_lag(df).dropna(subset=["pm25_ugm3", "target"])
X, y, w = tambah_fitur(d), d["target"], d["waktu"]

def latih(mask):
    m = HistGradientBoostingRegressor(random_state=42)
    m.fit(X[mask], y[mask])
    return m

def evaluasi(model, mask):
    pred = model.predict(X[mask])
    alarm = pred > AMBANG_ALARM
    kejadian = y[mask].to_numpy() > AMBANG_KEJADIAN
    return {
        "mae": round(float(mean_absolute_error(y[mask], pred)), 3),
        "recall": round(float(recall_score(kejadian, alarm, zero_division=0)), 3),
        "precision": round(float(precision_score(kejadian, alarm, zero_division=0)), 3),
        "f1": round(float(f1_score(kejadian, alarm, zero_division=0)), 3),
        "jumlah_jam": int(mask.sum()),
        "jumlah_kejadian_di_atas_55": int(kejadian.sum()),
        "jumlah_alarm": int(alarm.sum()),
        "alarm_benar": int((alarm & kejadian).sum()),
    }

# --- 1. Cek ulang validation dulu (test BELUM dibuka) ---
tr = w < "2026-03-01"
va = (w >= "2026-03-01") & (w < AWAL_TEST)
cek_val = evaluasi(latih(tr), va)
print("Cek validation:", cek_val)
if abs(cek_val["mae"] - 4.695) > 0.01:
    sys.exit("MAE validation beda dari 4.695 -> ada yang tidak sama. Berhenti, test belum dibuka.")

# --- 2. Latih model final di train + validation ---
# Baris 3 jam terakhir April dibuang karena targetnya sudah jatuh di bulan Mei (data test)
mask_final = w < AWAL_TEST - pd.Timedelta(hours=HORIZON_JAM)
model = latih(mask_final)

# --- 3. TEST: dibuka SATU KALI ---
te = (w >= AWAL_TEST) & (w < AKHIR_TEST)
metrik_test = evaluasi(model, te)
print("\n=== HASIL TEST (Mei-Juni 2026) ===")
for k, v in metrik_test.items():
    print(f"{k}: {v}")

# --- 4. Simpan model + metadata ---
data_latih = df[df["waktu"] < AWAL_TEST]
rentang_input = {
    k: {"min": float(data_latih[k].min()), "max": float(data_latih[k].max())}
    for k in ["pm25_ugm3", "suhu_C", "kelembaban_pct", "kecepatan_angin_ms",
              "arah_angin_deg", "curah_hujan_mm"]
}

(ROOT / "artifacts").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "artifacts" / "model.joblib")
meta = {
    "versi_model": "1.0.0",
    "dibuat": datetime.now().isoformat(timespec="seconds"),
    "algoritma": "HistGradientBoostingRegressor(random_state=42)",
    "target": f"PM2.5 (µg/m³) {HORIZON_JAM} jam ke depan",
    "horizon_jam": HORIZON_JAM,
    "fitur": FITUR,
    "stasiun": STASIUN,
    "ambang_alarm": AMBANG_ALARM,
    "ambang_kejadian": AMBANG_KEJADIAN,
    "periode_latih_final": f"awal data s.d. {(AWAL_TEST - pd.Timedelta(hours=HORIZON_JAM)).date()} (train + validation)",
    "periode_test": "2026-05-01 s.d. 2026-06-30",
    "metrik_validation": {**cek_val, "mae_baseline": 6.255},
    "metrik_test": metrik_test,
    "rentang_input_data_latih": rentang_input,
    "catatan": "Validation jatuh di musim hujan, test di awal kemarau. Batas alarm dipilih di validation, tidak diubah setelah test.",
    "versi_library": {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    },
}
META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nTersimpan: artifacts/model.joblib dan artifacts/metadata.json")