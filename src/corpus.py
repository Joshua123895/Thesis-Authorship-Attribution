"""Loader korpus terpusat.

Kenapa modul ini ada: sejak pandas 2.x/3.x, kalau pyarrow terpasang maka kolom
string dibaca sebagai ArrowStringArray, dan `.values` TIDAK mengembalikan numpy
array. scikit-learn tidak bisa meng-index tipe itu dan melempar

    TypeError: only integer scalar arrays can be converted to a scalar index

Semua script memuat data lewat sini supaya hasilnya selalu numpy array biasa,
terlepas dari ada atau tidaknya pyarrow di mesin yang menjalankan.
"""
import numpy as np
import pandas as pd

import config as C


def _to_numpy(series) -> np.ndarray:
    """Paksa jadi numpy object array, apa pun backend pandas-nya."""
    return np.asarray(series.astype(str).tolist(), dtype=object)


def load_docs():
    """Korpus agregat. Return: (texts, authors, topics, dataframe)."""
    df = pd.read_csv(C.DATA_PROC / "docs_agg.csv")
    return (_to_numpy(df["jawaban"]),
            _to_numpy(df["siswa"]),
            _to_numpy(df["topic"]),
            df)


def load_raw():
    """Jawaban mentah, satu jawaban per baris."""
    df = pd.read_csv(C.DATA_PROC / "answers_raw.csv")
    return (_to_numpy(df["jawaban"]),
            _to_numpy(df["siswa"]),
            _to_numpy(df["topic"]),
            df)
