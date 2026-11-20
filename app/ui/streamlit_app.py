"""Aplikasi Streamlit prakiraan PM2.5 Bandung untuk warga dan petugas dinas.

Jalankan dari root repo:
    streamlit run app/ui/streamlit_app.py

Dua mode (sama seperti template P3):
- Mode A (API_URL kosong): model di-load langsung di sini. Ini yang di-deploy ke Streamlit Cloud.
- Mode B (API_URL diisi): memanggil FastAPI lewat HTTP.
  Isi lewat environment variable API_URL, atau di Streamlit Cloud: Settings -> Secrets -> API_URL = "https://..."
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # Streamlit menjalankan file ini dari app/ui, jadi root repo ditambah manual
from app import predictor  # noqa: E402
from app.schema import InputData, terjemahkan_error  # noqa: E402
from src.features import STASIUN, buat_lag  # noqa: E402


def _api_url() -> str:
    try:
        url = st.secrets.get("API_URL", "")
    except Exception:  # tidak ada secrets.toml: wajar saat lokal
        url = ""
    return (url or os.getenv("API_URL", "")).rstrip("/")


API_URL = _api_url()
TIMEOUT = 60  # detik; API free tier yang sedang tidur butuh waktu untuk bangun


def prediksi(row: dict) -> tuple[dict | None, list[str]]:
    """(hasil, daftar_pesan_penolakan). Mode A dan Mode B memakai aturan validasi yang sama."""
    if API_URL:
        r = requests.post(API_URL + "/predict", json=row, timeout=TIMEOUT)
        if r.status_code == 422:
            return None, r.json().get("pesan", ["Input ditolak API (422)."])
        r.raise_for_status()
        return r.json(), []
    try:
        x = InputData(**row)
    except ValidationError as e:
        return None, terjemahkan_error(e.errors())
    return predictor.predict_one(x.model_dump()), []


def info_model() -> dict:
    if not API_URL:
        return predictor.load()[1]
    r = requests.get(API_URL + "/health", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ---------- Contoh dari data asli (untuk demo) ----------
KOLOM_FORM = ["stasiun", "tanggal", "jam", "pm25_ugm3", "pm25_lag1", "pm25_lag3", "pm25_lag24",
              "suhu_C", "kelembaban_pct", "kecepatan_angin_ms", "arah_angin_deg", "curah_hujan_mm"]
KOLOM_ANGKA = KOLOM_FORM[3:]


@st.cache_data
def data_contoh() -> pd.DataFrame | None:
    path = ROOT / "data" / "clean.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["waktu"])
    df["aktual_3j"] = df.groupby("stasiun")["pm25_ugm3"].shift(-3)  # nilai sebenarnya 3 jam kemudian
    d = buat_lag(df).dropna(subset=KOLOM_ANGKA + ["aktual_3j"])
    d = d[d["stasiun"].isin(STASIUN)]
    rentang = predictor.load()[1]["rentang_input_data_latih"]
    for kolom in KOLOM_ANGKA:  # ambil hanya baris yang lolos validasi
        r = rentang["pm25_ugm3" if kolom.startswith("pm25") else kolom]
        d = d[d[kolom].between(r["min"], r["max"])]
    return d


def isi_contoh(tinggi: bool):
    d = data_contoh()
    if tinggi:
        d = d[d["aktual_3j"] > 55]
    b = d.sample(1).iloc[0]
    st.session_state.update({
        "stasiun": b["stasiun"], "tanggal": b["waktu"].date(), "jam": b["waktu"].time(),
        **{k: float(b[k]) for k in KOLOM_ANGKA},
        "aktual_3j": float(b["aktual_3j"]),
    })


def kosongkan():
    for k in KOLOM_FORM + ["aktual_3j"]:
        st.session_state[k] = None


for k in KOLOM_FORM + ["aktual_3j"]:
    st.session_state.setdefault(k, None)


# ---------- Tampilan ----------
st.set_page_config(page_title="Prakiraan PM2.5 Bandung", page_icon="🌫️", layout="centered")
st.title("🌫️ Prakiraan PM2.5 Bandung")
st.caption("Isi kondisi saat ini di sebuah stasiun, dapatkan prakiraan PM2.5 **3 jam lagi** beserta saran.")

try:
    META = info_model()
except Exception as e:
    st.error(f"Model belum siap: {e}")
    st.stop()

mode = f"Mode B · memanggil API {API_URL}" if API_URL else "Mode A · model di-load langsung"
st.info(f"{mode} · versi model {META['versi_model']} · peringatan jika prakiraan > {META['ambang_alarm']} µg/m³")

if data_contoh() is not None:
    c1, c2, c3 = st.columns(3)
    c1.button("🎲 Contoh acak dari data", on_click=isi_contoh, args=(False,), use_container_width=True)
    c2.button("🔥 Contoh saat polusi tinggi", on_click=isi_contoh, args=(True,), use_container_width=True)
    c3.button("🧹 Kosongkan form", on_click=kosongkan, use_container_width=True)

with st.form("form_prediksi"):
    st.selectbox("Stasiun", STASIUN, key="stasiun", placeholder="Pilih stasiun")
    c1, c2 = st.columns(2)
    c1.date_input("Tanggal pengukuran terakhir", key="tanggal")
    c2.time_input("Jam pengukuran terakhir (WIB)", key="jam", step=3600)

    st.markdown("**PM2.5 (µg/m³)**")
    c1, c2, c3, c4 = st.columns(4)
    c1.number_input("Sekarang", key="pm25_ugm3", step=1.0)
    c2.number_input("1 jam lalu", key="pm25_lag1", step=1.0)
    c3.number_input("3 jam lalu", key="pm25_lag3", step=1.0)
    c4.number_input("24 jam lalu", key="pm25_lag24", step=1.0)

    st.markdown("**Cuaca saat ini (pada tanggal & jam pengukuran)**")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Suhu (°C)", key="suhu_C", step=0.5)
    c2.number_input("Kelembapan (%)", key="kelembaban_pct", step=1.0)
    c3.number_input("Kecepatan angin (m/s)", key="kecepatan_angin_ms", step=0.1)
    c1, c2 = st.columns(2)
    c1.number_input("Arah angin (°)", key="arah_angin_deg", step=1.0,
                    help="0 = utara, 90 = timur, 180 = selatan, 270 = barat")
    c2.number_input("Curah hujan (mm)", key="curah_hujan_mm", step=0.1)

    kirim = st.form_submit_button("Prakirakan", type="primary", use_container_width=True)

if kirim:
    s = st.session_state
    label = {"stasiun": "Stasiun", "tanggal": "Tanggal", "jam": "Jam", "pm25_ugm3": "PM2.5 sekarang",
             "pm25_lag1": "PM2.5 1 jam lalu", "pm25_lag3": "PM2.5 3 jam lalu", "pm25_lag24": "PM2.5 24 jam lalu",
             "suhu_C": "Suhu", "kelembaban_pct": "Kelembapan", "kecepatan_angin_ms": "Kecepatan angin",
             "arah_angin_deg": "Arah angin", "curah_hujan_mm": "Curah hujan"}
    kosong = [label[k] for k in KOLOM_FORM if s[k] is None]
    if kosong:
        st.error("**Input ditolak.** Belum diisi: " + ", ".join(kosong))
        st.stop()

    row = {"stasiun": s["stasiun"], "waktu": datetime.combine(s["tanggal"], s["jam"]).isoformat(),
           **{k: s[k] for k in KOLOM_ANGKA}}
    try:
        with st.spinner("Menghitung prakiraan..."):
            hasil, ditolak = prediksi(row)
    except Exception as e:
        st.error(f"Gagal memprediksi: {e}")
        st.stop()

    if ditolak:
        st.error("**Input ditolak:**\n\n" + "\n".join(f"- {p}" for p in ditolak))
        st.stop()

    waktu = datetime.fromisoformat(hasil["waktu_prediksi"])
    st.subheader(f"{hasil['stasiun']} · {waktu:%d-%m-%Y} pukul {waktu:%H:%M}")
    c1, c2 = st.columns(2)
    c1.metric("Prakiraan PM2.5", f"{hasil['prediksi_pm25_ugm3']} µg/m³")
    c2.metric("Kategori", hasil["kategori"])

    if hasil["peringatan"]:
        st.warning(f"⚠️ **PERINGATAN** · prakiraan di atas {hasil['ambang_alarm']} µg/m³\n\n{hasil['saran']}")
    else:
        st.success(f"Tidak ada peringatan\n\n{hasil['saran']}")

    if s["aktual_3j"] is not None:
        st.caption(f"Contoh dari data: nilai PM2.5 sebenarnya 3 jam kemudian = **{s['aktual_3j']:.1f} µg/m³**")

    with st.expander("Lihat request & response JSON"):
        st.json({"request": row, "response": hasil})

with st.expander("Tentang model"):
    t = META["metrik_test"]
    st.markdown(f"""
- Versi model {META['versi_model']} · prakiraan {META['horizon_jam']} jam ke depan
- Rata-rata kesalahan di data test (Mei–Juni 2026): ±{t['mae']} µg/m³ (baseline *persistence* lebih buruk)
- Peringatan muncul jika prakiraan > {META['ambang_alarm']} µg/m³. Di data test, peringatan menangkap
  {t['recall']:.0%} jam dengan PM2.5 > 55, dan {t['precision']:.0%} peringatan terbukti benar.
- Kategori (Baik / Sedang / Tidak sehat) dan saran hanya label tampilan untuk nilai prakiraan.
""")
