"""Aplikasi Streamlit prediksi PM2.5.
Jalankan dari folder repo:  streamlit run app/app.py
(API harus sudah jalan: uvicorn api.main:app --reload)
"""
import json, os, sys
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src.features import STASIUN

META = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))

def ambil_api_url():
    try:
        return st.secrets["API_URL"]          # dipakai saat deploy
    except Exception:
        return os.environ.get("API_URL", "http://localhost:8000")   # saat di laptop

API_URL = ambil_api_url().rstrip("/")

st.set_page_config(page_title="Prediksi PM2.5 Bandung", page_icon="🌫️")
st.title("🌫️ Prediksi PM2.5 Bandung")
st.caption(f"Memprediksi konsentrasi PM2.5 **{META['horizon_jam']} jam ke depan** "
           "berdasarkan kondisi saat ini.")

with st.form("form_prediksi"):
    stasiun = st.selectbox("Stasiun", STASIUN, index=None, placeholder="Pilih stasiun")
    c1, c2 = st.columns(2)
    tanggal = c1.date_input("Tanggal pengukuran", value=None)
    jam = c2.time_input("Jam pengukuran", value=None, step=3600)

    st.markdown("**PM2.5 (µg/m³)**")
    c1, c2, c3, c4 = st.columns(4)
    pm_now = c1.number_input("Sekarang", value=None, step=1.0)
    pm_1 = c2.number_input("1 jam lalu", value=None, step=1.0)
    pm_3 = c3.number_input("3 jam lalu", value=None, step=1.0)
    pm_24 = c4.number_input("24 jam lalu", value=None, step=1.0)

    st.markdown("**Cuaca**")
    c1, c2, c3 = st.columns(3)
    suhu = c1.number_input("Suhu (°C)", value=None, step=0.5)
    lembap = c2.number_input("Kelembapan (%)", value=None, step=1.0)
    angin = c3.number_input("Kecepatan angin (m/s)", value=None, step=0.1)
    c1, c2 = st.columns(2)
    arah = c1.number_input("Arah angin (°)", value=None, step=1.0,
                           help="0 = utara, 90 = timur, 180 = selatan, 270 = barat")
    hujan = c2.number_input("Curah hujan (mm)", value=None, step=0.1)

    kirim = st.form_submit_button("Prediksi", type="primary")

if kirim:
    # 1) Tolak isian kosong
    isian = {"Stasiun": stasiun, "Tanggal": tanggal, "Jam": jam,
             "PM2.5 sekarang": pm_now, "PM2.5 1 jam lalu": pm_1,
             "PM2.5 3 jam lalu": pm_3, "PM2.5 24 jam lalu": pm_24,
             "Suhu": suhu, "Kelembapan": lembap, "Kecepatan angin": angin,
             "Arah angin": arah, "Curah hujan": hujan}
    kosong = [k for k, v in isian.items() if v is None]
    if kosong:
        st.error("**Input ditolak.** Belum diisi: " + ", ".join(kosong))
        st.stop()

    # 2) Kirim ke API (API yang menolak nilai mustahil / di luar rentang)
    data = {
        "stasiun": stasiun,
        "waktu": datetime.combine(tanggal, jam).isoformat(),
        "pm25_ugm3": pm_now, "pm25_lag1": pm_1, "pm25_lag3": pm_3, "pm25_lag24": pm_24,
        "suhu_C": suhu, "kelembaban_pct": lembap, "kecepatan_angin_ms": angin,
        "arah_angin_deg": arah, "curah_hujan_mm": hujan,
    }
    try:
        with st.spinner("Menghitung prediksi..."):
            r = requests.post(f"{API_URL}/predict", json=data, timeout=90)
    except requests.RequestException:
        st.error("Tidak bisa terhubung ke API. Pastikan API sedang berjalan, lalu coba lagi.")
        st.stop()

    if r.status_code == 422:
        pesan = r.json().get("pesan", [])
        st.error("**Input ditolak:**\n\n" + "\n".join(f"- {p}" for p in pesan))
        st.stop()
    if r.status_code != 200:
        st.error(f"Terjadi kesalahan di server (kode {r.status_code}).")
        st.stop()

    # 3) Tampilkan hasil
    h = r.json()
    waktu_pred = datetime.fromisoformat(h["waktu_prediksi"])
    st.subheader(f"Prediksi untuk {h['stasiun']}, {waktu_pred:%d-%m-%Y pukul %H:%M}")
    c1, c2 = st.columns(2)
    c1.metric("Prediksi PM2.5", f"{h['prediksi_pm25_ugm3']} µg/m³")
    c2.metric("Kategori", h["kategori"])

    if h["peringatan"]:
        st.warning(f"⚠️ **PERINGATAN:** PM2.5 diprediksi di atas {h['ambang_alarm']} µg/m³. "
                   "Kurangi aktivitas di luar ruangan dan gunakan masker.")
    else:
        st.success(f"Tidak ada peringatan (prediksi tidak melebihi {h['ambang_alarm']} µg/m³).")

with st.expander("Tentang model"):
    t = META["metrik_test"]
    st.markdown(f"""
- Model: {META['algoritma']} — versi {META['versi_model']}
- Rata-rata kesalahan di data test (Mei–Juni 2026): ±{t['mae']} µg/m³
- Peringatan muncul jika prediksi > {META['ambang_alarm']} µg/m³.
  Di data test, peringatan menangkap {t['recall']:.0%} jam dengan PM2.5 > 55,
  dan {t['precision']:.0%} peringatan terbukti benar.
- Kategori (Baik / Sedang / Tidak sehat) hanya label tampilan untuk nilai prediksi.
""")