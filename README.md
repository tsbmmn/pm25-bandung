# Prakiraan PM2.5 Bandung · Tim [nama tim]

**Kasus:** 06 · Prakiraan PM2.5 Bandung · **Aplikasi:** [tautan Streamlit] · **Video:** [tautan video]

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| Denia [nama lengkap] | [NIM] | EDA, pembersihan data, baseline, fitur, pemilihan model dan batas alarm |
| Nayla Putri Ashfiya | [NIM] | model final, evaluasi test, artifact + metadata, API FastAPI, aplikasi Streamlit, deploy |
| [nama] | [NIM] | [bagian] |

## Masalah dan pengguna

Tim kesehatan lingkungan dinas kota ingin memperingatkan warga **sebelum** udara memburuk. Model ini
memprakirakan PM2.5 **tiga jam ke depan** di lima stasiun (Antapani, Cibeureum, Dago, Soekarno-Hatta,
Ujungberung) dari kondisi sampai jam saat ini, sehingga dinas bisa mengumumkan peringatan lebih awal dan warga
sensitif (misalnya penderita asma) bisa bersiap memakai masker.

## Data dan pembersihan

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| [isi dari notebooks/01_eda.ipynb] | | |

## Hasil

Pembagian data berdasarkan waktu: *train* s.d. Februari 2026, *validation* Maret–April 2026, *test* Mei–Juni 2026
(dibuka satu kali, lewat `scripts/train_final.py`). Model final dilatih ulang pada *train* + *validation*.

| Model | MAE *validation* (µg/m³) | MAE *test* (µg/m³) |
|---|---|---|
| *Baseline*: persistence (3 jam lagi = sekarang) | 6,255 | [metrik_baseline.test.mae] |
| Ridge | 5,113 | – |
| **HistGradientBoostingRegressor** (final) | **4,695** | **4,683** |

Peringatan jam dengan PM2.5 > 55 µg/m³ ("tidak sehat"):

| | Recall *validation* | Precision *validation* | Recall *test* | Precision *test* |
|---|---|---|---|---|
| HistGradientBoosting, alarm jika prakiraan > 45 | 0,465 | 0,328 | 0,736 | 0,339 |

*Threshold* yang dipakai: **45 µg/m³** pada prakiraan, dipilih di *validation* karena F1-nya tertinggi (0,385).
Sengaja lebih rendah dari 55 agar warga diperingatkan sebelum udara benar-benar tidak sehat; konsekuensinya
sekitar dua dari tiga peringatan adalah alarm palsu.

## Cara menjalankan

```bash
pip install -r requirements-dev.txt
pytest                                  # smoke test API: normal, ekstrem, tidak wajar
uvicorn app.api.main:app --reload       # API  -> http://127.0.0.1:8000/docs
streamlit run app/ui/streamlit_app.py   # aplikasi (Mode A: model di-load langsung)
```

Mode B (aplikasi memanggil API): jalankan API, lalu `API_URL=http://127.0.0.1:8000 streamlit run app/ui/streamlit_app.py`
(PowerShell: `$env:API_URL="http://127.0.0.1:8000"` dulu).

| Endpoint | Fungsi |
|---|---|
| `GET /health` | cek server hidup + versi model, batas alarm, rentang input |
| `POST /predict` | prakiraan satu stasiun |
| `POST /predict/batch` | beberapa baris sekaligus (maks 1000) |
| `GET /docs` | dokumentasi interaktif |

Masukan yang ditolak (422): kolom kosong, bukan angka/NaN, kolom asing, stasiun tak dikenal, nilai mustahil
(PM2.5 negatif, kelembapan > 100%, arah angin > 360°), dan nilai di luar rentang data latih.

## Struktur repositori

```
notebooks/            01_eda (pembersihan, EDA, baseline), 02_model (fitur, model, batas alarm)
data/                 raw.csv dan clean.csv
src/features.py       fungsi pembuat fitur, dipakai saat pelatihan DAN saat prediksi
scripts/              train_final.py (model final + test sekali), lengkapi_metadata.py (baseline + fitur_input)
artifacts/            model.joblib + metadata.json
app/predictor.py      muat model sekali, lalu prediksi (dipakai API dan aplikasi)
app/schema.py         aturan validasi masukan (dipakai API dan aplikasi)
app/api/main.py       FastAPI
app/ui/streamlit_app.py  aplikasi Streamlit
tests/test_api.py     smoke test
```

## Keterbatasan

- *Validation* jatuh di musim hujan dan *test* di awal kemarau; hanya 43 jam kejadian > 55 di *validation*, jadi
  angka recall/precision di sana mudah bergoyang.
- Precision peringatan sekitar 34%: sebagian besar peringatan adalah alarm palsu.
- Masukan di luar rentang data latih ditolak karena model tidak bisa dipercaya di sana (misalnya kejadian polusi
  ekstrem yang belum pernah terjadi di data).
- Model butuh riwayat PM2.5 (1, 3, dan 24 jam lalu); jika sensor mati, prakiraan tidak bisa dibuat.

## Daftar periksa sebelum dikumpulkan

- [ ] Repositori bersifat publik dan setiap anggota memiliki *commit*
- [ ] *Notebook* dapat dijalankan ulang dari awal sampai akhir tanpa galat
- [ ] *Baseline* wajib dilaporkan dan dibandingkan
- [x] Data *test* hanya dibuka satu kali, di akhir
- [x] `artifacts/metadata.json` berisi versi model, fitur masukan, *threshold*, dan metrik
- [ ] Aplikasi dapat dibuka orang lain melalui tautan, termasuk dari ponsel
- [x] Aplikasi tidak berhenti mendadak ketika menerima masukan yang tidak wajar
- [ ] Laporan paling banyak 6 halaman dan video paling lama 5 menit
