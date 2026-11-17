"""Ekstraksi fitur.

Dua blok:
  1. char n-gram TF-IDF  -> fitur utama, membawa hampir seluruh sinyal
  2. stilometri          -> pendukung, HARUS dibobot kecil saat digabung

Catatan penting: menggabungkan kedua blok tanpa pembobotan akan MERUSAK hasil.
Setelah StandardScaler, magnitudo blok stilometri jauh lebih besar daripada
nilai TF-IDF (0-1), sehingga mendominasi dan menenggelamkan char n-gram.
"""
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize

import config as C


def stylometric_vector(text: str) -> list:
    """245 fitur stilometri untuk satu dokumen, dinormalisasi panjang.

    Yule's K sengaja TIDAK dipakai: butuh ribuan kata agar stabil, sedangkan
    dokumen di korpus ini median 200 kata.
    """
    words = text.split()
    n = max(len(words), 1)
    chars = max(len(text), 1)
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]

    wlen = [len(w) for w in words] or [0]
    slen = [len(s.split()) for s in sents] or [0]
    lw = [w.strip(".,;:!?()\"'").lower() for w in words]
    counts = pd.Series(lw).value_counts()

    feats = [
        np.mean(wlen), np.std(wlen),
        np.mean(slen), np.std(slen),
        len(set(lw)) / n,                                  # type-token ratio
        (counts == 1).sum() / n,                           # hapax ratio
        sum(c.isupper() for c in text) / chars,
        sum(c.isdigit() for c in text) / chars,
    ]
    feats += [text.count(p) / chars * 100 for p in C.PUNCT]
    feats += [counts.get(w, 0) / n * 100 for w in C.FUNCTION_WORDS]
    return feats


def stylometric_matrix(texts) -> np.ndarray:
    return np.asarray([stylometric_vector(t) for t in texts], dtype=float)


def length_matrix(texts) -> np.ndarray:
    return np.asarray([[len(t.split())] for t in texts], dtype=float)


def build(texts_train, texts_test, mode: str,
          stylo_train=None, stylo_test=None, weight: float = None):
    """Bangun matriks fitur train/test.

    mode: "char" | "stylo" | "length" | "hybrid"
    Vectoriser dan scaler SELALU di-fit pada training saja.
    """
    weight = C.STYLO_WEIGHT if weight is None else weight
    parts_tr, parts_te = [], []

    if mode in ("char", "hybrid"):
        vec = TfidfVectorizer(**C.CHAR_NGRAM)
        parts_tr.append(vec.fit_transform(texts_train))
        parts_te.append(sparse.csr_matrix(vec.transform(texts_test)))

    if mode in ("stylo", "hybrid"):
        scaler = StandardScaler().fit(stylo_train)
        a = normalize(scaler.transform(stylo_train))
        b = normalize(scaler.transform(stylo_test))
        if mode == "hybrid":                 # blok pendukung -> dibobot kecil
            a, b = a * weight, b * weight
        parts_tr.append(sparse.csr_matrix(a))
        parts_te.append(sparse.csr_matrix(b))

    if mode == "length":
        scaler = StandardScaler().fit(stylo_train)
        parts_tr.append(sparse.csr_matrix(scaler.transform(stylo_train)))
        parts_te.append(sparse.csr_matrix(scaler.transform(stylo_test)))

    return sparse.hstack(parts_tr).tocsr(), sparse.hstack(parts_te).tocsr()
