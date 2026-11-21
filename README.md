# Prakiraan PM2.5 Bandung · Tim [Anak Bang Rifki]

**Kasus:** 06 · Prakiraan PM2.5 Bandung · **Aplikasi:** [https://pm25-bandung-82cg8k7i4ii6bvn48znlzf.streamlit.app/] · **Video:** [https://drive.google.com/file/d/1CJSCIG0lTCJuABX38eOTO6tqdzyYjrL_/view?usp=sharing]

| Anggota | NIM | Bagian yang dikerjakan |
|---|---|---|
| Teresia Denia Primisani | 2404107 | EDA, pembersihan data, baseline, fitur, pemilihan model dan batas alarm, deploy|
| Nayla Putri Ashfiya | 2404688 | model final, evaluasi test, artifact + metadata, API FastAPI, aplikasi Streamlit |

## Masalah dan pengguna

Tim kesehatan lingkungan dinas kota ingin memperingatkan warga sebelum udara memburuk. Model ini memprakirakan PM2.5 tiga jam ke depan di lima stasiun (Antapani, Cibeureum, Dago, Soekarno-Hatta, Ujungberung) dari kondisi sampai jam saat ini, sehingga dinas bisa mengumumkan peringatan lebih awal dan warga sensitif (misalnya penderita asma) bisa bersiap memakai masker.

Pengguna aplikasi adalah warga dan petugas dinas. Masukan: stasiun dan kondisi saat ini. Keluaran: prakiraan PM2.5 tiga jam lagi, kategorinya, dan saran.

## Data dan pembersihan

Data: `data/raw.csv`, satu baris adalah satu stasiun pada satu jam (Juli 2024 – Juni 2026), 87.650 baris mentah. Data sintetis dan belum dibersihkan. Hasil pembersihan: `data/clean.csv` (87.600 baris = 5 stasiun × 17.520 jam). Detail dan kodenya ada di `notebooks/01_eda.ipynb`.

| Masalah yang ditemukan | Jumlah baris | Cara menangani |
|---|---|---|
| Nama stasiun tidak konsisten (16 penulisan untuk 5 stasiun: huruf besar/kecil, spasi, "Soekarno Hatta", "SOETTA", "Ujung Berung") | 2.635 | dirapikan ke 5 nama baku |
| PM2.5 = -999 (kode gagal baca sensor) | 173 | diganti NaN, baris dipertahankan supaya urutan jam utuh |
| PM2.5 = 0,0 (nilai terkecil yang sah adalah 1,0; tetangga jamnya normal 20–45 µg/m³) | 87 | diganti NaN |
| Baris ganda (stasiun dan jam sama) | 405 | dibuang; 398 salinan persis |
| Tujuh pasangan baris Soekarno-Hatta (30 Juni 2025 pukul 17–23) berjam sama tetapi nilainya berbeda | 7 | baris pertama dipertahankan karena nyambung dengan jam sebelum dan sesudahnya; baris kedua mirip data 30 September 2025 pukul 17–23 yang hilang (dugaan salah label bulan), jam itu dibiarkan NaN |
| Jam hilang (tidak ada barisnya) | 355 | ditambahkan sebagai baris kosong (NaN), 344 blok 1 jam, 2 blok 2 jam, 1 blok 7 jam, sehingga `shift(-3)` selalu tepat tiga jam |
| Nilai kosong pada data asli: PM2.5, suhu, dan kelembapan (144 baris), kecepatan angin (877 baris) | 144 / 877 | dibiarkan NaN, ditangani model; tidak diinterpolasi karena interpolasi memakai nilai jam sesudahnya |
 
Yang sudah dicek dan tidak bermasalah: zona waktu (suhu terpanas sekitar pukul 15 di semua stasiun dan kuartal, jadi data sudah WIB dan tidak digeser), sensor macet (PM2.5 identik 6 jam berturut-turut: 0 kasus), dan rentang nilai kolom cuaca. Setelah pembersihan, PM2.5 kosong sebanyak 757 dari 87.600 baris (0,9%).

## Temuan EDA
 
- Soekarno-Hatta paling tinggi (rata-rata 34,5 µg/m³, maksimum 128,4) dan Dago paling rendah (rata-rata 18,9). Jam >55 hanya 2,75% dari seluruh data, dan didominasi Soekarno-Hatta dan Cibeureum.
- Ada pola musiman: PM2.5 tinggi sekitar Juni–Oktober dan rendah sekitar Desember–April.
- Pola harian: puncak sekitar pukul 07–08, terendah sekitar pukul 14–15. Penurunan sesaat pada pukul 06 terlihat di kelima stasiun, jadi ada di data, bukan galat satu sensor.
- 25 lonjakan lebih dari 40 µg/m³ dalam satu jam (0,03% data), semuanya bertahan beberapa jam sehingga dianggap kejadian asli dan dipertahankan.

## Hasil

Pembagian data berdasarkan waktu: *train* s.d. Februari 2026, *validation* Maret–April 2026, *test* Mei–Juni 2026 (dibuka satu kali, lewat `scripts/train_final.py`). Model final dilatih ulang pada *train* + *validation*.
 
Fitur (hanya informasi sampai jam saat ini): PM2.5 sekarang, PM2.5 1, 3, dan 24 jam lalu, tren 3 jam, jam (sin/cos), hari, suhu, kelembapan, kecepatan angin, arah angin (sin/cos), curah hujan, dan stasiun. Semuanya dibuat di `src/features.py`.
 
| Model | MAE *validation* (µg/m³) | MAE *test* (µg/m³) |
|---|---|---|
| *Baseline*: persistence (3 jam lagi = sekarang) | 6,255 | [isi dari artifacts/metadata.json; di notebook: 6,500] |
| Ridge | 5,113 | – |
| **HistGradientBoostingRegressor** (final) | **4,695** | **4,683** |
 
MAE *validation* per stasiun (model dilatih pada *train* saja):
 
| Stasiun | Model | Baseline |
|---|---|---|
| Antapani | 4,39 | 5,86 |
| Cibeureum | 5,16 | 6,94 |
| Dago | 3,44 | 4,32 |
| Soekarno-Hatta | 6,35 | 8,87 |
| Ujungberung | 4,15 | 5,30 |
 
Peringatan jam dengan PM2.5 > 55 µg/m³ ("tidak sehat"):
 
| | Recall *validation* | Precision *validation* | Recall *test* | Precision *test* |
|---|---|---|---|---|
| HistGradientBoosting, alarm jika prakiraan > 45 | 0,465 | 0,328 | 0,736 | 0,339 |
 
*Threshold* yang dipakai: **45 µg/m³** pada prakiraan, dipilih di *validation* karena F1-nya tertinggi (0,385). Sengaja lebih rendah dari 55 agar warga diperingatkan sebelum udara benar-benar tidak sehat; konsekuensinya sekitar dua dari tiga peringatan adalah alarm palsu. Jam >55 berjumlah 43 di *validation* dan 239 di *test*.
 
Beberapa ide perbaikan sudah dicoba di *validation* (target berupa selisih, fitur turunan, klasifikator khusus >55) dan hasilnya hampir tidak berbeda dari model final, jadi model tidak diubah setelah *test* dibuka.

## Cara menjalankan

Python 3.14.3 (versi yang dipakai saat pelatihan; versi library ada di `artifacts/metadata.json`).
 
```
pip install -r requirements-dev.txt
pytest                                  # smoke test API: normal, ekstrem, tidak wajar
uvicorn app.api.main:app --reload       # API  -> http://127.0.0.1:8000/docs
streamlit run app/ui/streamlit_app.py   # aplikasi (Mode A: model di-load langsung)
```
 
Mode B (aplikasi memanggil API): jalankan API, lalu `API_URL=http://127.0.0.1:8000 streamlit run app/ui/streamlit_app.py` (PowerShell: `$env:API_URL="http://127.0.0.1:8000"` dulu).
 
| Endpoint | Fungsi |
|---|---|
| `GET /health` | cek server hidup + versi model, batas alarm, rentang input |
| `POST /predict` | prakiraan satu stasiun |
| `POST /predict/batch` | beberapa baris sekaligus (maks 1000) |
| `GET /docs` | dokumentasi interaktif |
 
Masukan yang ditolak (422): kolom kosong, bukan angka/NaN, kolom asing, stasiun tak dikenal, nilai mustahil (PM2.5 negatif, kelembapan > 100%, arah angin > 360°), dan nilai di luar rentang data latih.

## Deploy
 
Aplikasi di-deploy di Streamlit Community Cloud (Mode A, model dimuat langsung, `API_URL` kosong) dengan *main file* `app/ui/streamlit_app.py`. `requirements.txt` hanya berisi paket yang dipakai aplikasi, dengan versi yang sama seperti saat pelatihan. Paket untuk pengembangan (Jupyter, pengujian, dan seterusnya) ada di `requirements-dev.txt`.
 
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
 
- *Validation* jatuh di musim hujan dan *test* di awal kemarau; hanya 43 jam kejadian > 55 di *validation*, jadi angka recall/precision di sana mudah bergoyang. Recall *test* yang lebih tinggi tidak diklaim sebagai tanda model membaik.
- Precision peringatan sekitar 34%: sebagian besar peringatan adalah alarm palsu.
- Masukan di luar rentang data latih ditolak karena model tidak bisa dipercaya di sana (misalnya kejadian polusi ekstrem yang belum pernah terjadi di data).
- Model butuh riwayat PM2.5 (1, 3, dan 24 jam lalu); jika sensor mati, prakiraan tidak bisa dibuat.
- Batas antar-periode (*train*/*validation*/*test*) tidak diberi jeda tiga jam, sehingga sekitar 15 baris terakhir tiap periode memiliki target yang jatuh di periode berikutnya; dampaknya sangat kecil (sekitar 0,02% baris latih).
- Data bersifat sintetis; model ini bukan alat kesehatan dan tidak menggantikan pengukuran resmi.


## Daftar periksa sebelum dikumpulkan

- [x] Repositori bersifat publik dan setiap anggota memiliki *commit*
- [x] *Notebook* dapat dijalankan ulang dari awal sampai akhir tanpa galat
- [x] *Baseline* wajib dilaporkan dan dibandingkan
- [x] Data *test* hanya dibuka satu kali, di akhir
- [x] `artifacts/metadata.json` berisi versi model, fitur masukan, *threshold*, dan metrik
- [x] Aplikasi dapat dibuka orang lain melalui tautan, termasuk dari ponsel
- [x] Aplikasi tidak berhenti mendadak ketika menerima masukan yang tidak wajar
- [ ] Laporan paling banyak 6 halaman dan video paling lama 5 menit
