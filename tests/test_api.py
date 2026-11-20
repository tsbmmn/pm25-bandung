"""Smoke test API: input normal, ekstrem, dan tidak wajar. Jalankan dari root repo: pytest -q"""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.schema import CONTOH as NORMAL


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` menjalankan startup: model di-load sekali
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_normal(client):
    r = client.post("/predict", json=NORMAL)
    assert r.status_code == 200, r.text
    h = r.json()
    assert h["prediksi_pm25_ugm3"] >= 0
    assert h["kategori"] in ("Baik", "Sedang", "Tidak sehat")
    assert h["peringatan"] == (h["prediksi_pm25_ugm3"] > h["ambang_alarm"])
    assert h["waktu_prediksi"].startswith("2026-06-15T10:00")  # 07.00 + 3 jam
    assert h["saran"]


def test_ekstrem_tapi_masih_dalam_rentang(client):
    r = client.get("/health").json()["rentang_input_data_latih"]
    ekstrem = {**NORMAL, "pm25_ugm3": r["pm25_ugm3"]["max"], "pm25_lag1": r["pm25_ugm3"]["max"],
               "kecepatan_angin_ms": r["kecepatan_angin_ms"]["min"], "curah_hujan_mm": 0}
    assert client.post("/predict", json=ekstrem).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("pm25_ugm3", -5),            # mustahil
    ("kelembaban_pct", 150),      # mustahil
    ("arah_angin_deg", 400),      # mustahil
    ("stasiun", "Bogor"),         # stasiun tidak dikenal
    ("pm25_ugm3", "abc"),         # bukan angka
    ("pm25_ugm3", None),          # kosong
    ("waktu", "kemarin"),         # format waktu salah
    ("pm25_ugm3", 999),           # mungkin secara fisik, tapi di luar rentang data latih
])
def test_tidak_wajar_ditolak_422(client, field, value):
    r = client.post("/predict", json={**NORMAL, field: value})
    assert r.status_code == 422
    assert r.json()["ditolak"] is True and r.json()["pesan"]


def test_kolom_hilang_ditolak(client):
    body = {k: v for k, v in NORMAL.items() if k != "suhu_C"}
    r = client.post("/predict", json=body)
    assert r.status_code == 422 and "suhu_C: wajib diisi" in r.json()["pesan"]


def test_kolom_asing_ditolak(client):
    assert client.post("/predict", json={**NORMAL, "kolom_aneh": 1}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json=[NORMAL, {**NORMAL, "stasiun": "Soekarno-Hatta"}])
    assert r.status_code == 200 and len(r.json()) == 2
    assert client.post("/predict/batch", json=[]).json() == []
