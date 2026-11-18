import numpy as np
import pandas as pd

STASIUN = ["Antapani", "Cibeureum", "Dago", "Soekarno-Hatta", "Ujungberung"]
FITUR = ["pm25_ugm3", "pm25_lag1", "pm25_lag3", "pm25_lag24", "tren_3j", "jam_sin", "jam_cos",
         "hari", "suhu_C", "kelembaban_pct", "kecepatan_angin_ms", "arah_sin", "arah_cos",
         "curah_hujan_mm", "stasiun_kode"]

def buat_lag(df):
    """Untuk pelatihan: hitung lag dari riwayat. Aplikasi mengisi lag dari input pengguna."""
    df = df.sort_values(["stasiun", "waktu"]).copy()
    g = df.groupby("stasiun")["pm25_ugm3"]
    for n in (1, 3, 24):
        df[f"pm25_lag{n}"] = g.shift(n)
    return df

def tambah_fitur(df):
    """Dipakai notebook DAN aplikasi: satu fungsi, dua pengguna."""
    df = df.copy()
    w = pd.to_datetime(df["waktu"])
    df["jam_sin"], df["jam_cos"] = np.sin(2*np.pi*w.dt.hour/24), np.cos(2*np.pi*w.dt.hour/24)
    df["hari"] = w.dt.dayofweek
    df["tren_3j"] = df["pm25_ugm3"] - df["pm25_lag3"]
    df["arah_sin"], df["arah_cos"] = np.sin(np.radians(df["arah_angin_deg"])), np.cos(np.radians(df["arah_angin_deg"]))
    df["stasiun_kode"] = df["stasiun"].map({s: i for i, s in enumerate(STASIUN)})
    return df[FITUR]
