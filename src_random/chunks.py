"""Korpus chunk campuran-topik + skema split acak.

Desain (menggantikan satu-dokumen-per-topik):
  1. 40 soal (4 topik x 10 soal) dibagi ACAK ke 4 kelompok soal, distratifikasi
     per topik: tiap kelompok berisi 10 soal, 2-3 soal dari tiap topik.
  2. Satu dokumen = jawaban satu siswa untuk satu kelompok soal
     -> 50 siswa x 4 kelompok = 200 dokumen, tiap dokumen mencampur 4 topik.
  3. Fold k = semua dokumen kelompok k sebagai data uji, sisanya data latih.
     Tiap siswa menyumbang 3 dokumen latih dan 1 dokumen uji per fold.

Kenapa kelompok soal dan bukan sekadar mengocok jawaban tiap siswa:
  kalau jawaban tiap siswa dikocok sendiri-sendiri, jawaban seorang siswa untuk
  soal q hanya ada di dokumen ujinya, sementara sebagian besar siswa lain punya
  jawaban soal q di data latih. Soal yang sama -> isi mirip -> prediksi tertarik
  ke siswa lain (jebakan yang sama dengan random split lama, tapi di level soal).
  Dengan kelompok soal yang sama untuk semua siswa, soal uji tidak pernah muncul
  di data latih siapa pun, dan komposisi topik latih/uji seimbang.

`naive_chunks` disediakan HANYA sebagai pembanding (pengocokan per siswa).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config as C                      # noqa: E402
from corpus import _to_numpy            # noqa: E402

N_GROUPS = 4


def load_answers():
    """Jawaban mentah, hanya 50 siswa yang lengkap 4 topik (1988 jawaban)."""
    raw = pd.read_csv(C.DATA_PROC / "answers_raw.csv")
    agg = pd.read_csv(C.DATA_PROC / "docs_agg.csv")
    raw = raw[raw["siswa"].astype(str).isin(agg["siswa"].astype(str).unique())].copy()
    raw["siswa"] = raw["siswa"].astype(str)
    return raw.reset_index(drop=True)


def question_groups(seed: int) -> dict:
    """Peta (topic, soal) -> kelompok 0..3, acak dan terstratifikasi per topik."""
    rng = np.random.default_rng(seed)
    order = []
    for topic in sorted(C.TOPICS):
        qs = np.arange(1, C.N_QUESTIONS + 1)
        rng.shuffle(qs)
        order += [(topic, int(q)) for q in qs]
    # urutan berantai lintas topik -> tiap kelompok tepat 10 soal, 2-3 per topik
    return {tq: i % N_GROUPS for i, tq in enumerate(order)}


def build_chunks(raw: pd.DataFrame, seed: int):
    """Return (texts, authors, groups, df) untuk satu pengacakan kelompok soal."""
    gmap = question_groups(seed)
    raw = raw.copy()
    raw["grp"] = [gmap[(t, int(q))] for t, q in zip(raw["topic"], raw["soal"])]
    df = (raw.sort_values(["siswa", "grp", "topic", "soal"])
             .groupby(["siswa", "grp"])
             .agg(jawaban=("jawaban", " ".join),
                  n_jawaban=("jawaban", "size"),
                  n_topik=("topic", "nunique"))
             .reset_index())
    df["n_kata"] = df["jawaban"].str.split().str.len()
    return (_to_numpy(df["jawaban"]), _to_numpy(df["siswa"]),
            df["grp"].to_numpy(), df)


def group_folds(groups):
    """Fold k: dokumen kelompok soal k = uji."""
    groups = np.asarray(groups)
    return [(np.where(groups != g)[0], np.where(groups == g)[0], f"grp{g+1}")
            for g in range(N_GROUPS)]


def naive_chunks(raw: pd.DataFrame, seed: int):
    """PEMBANDING: jawaban tiap siswa dikocok sendiri (terstratifikasi topik)
    lalu dibagi ke 4 chunk; split = StratifiedKFold per penulis."""
    rng = np.random.default_rng(seed)
    rows = []
    for s, sub in raw.groupby("siswa"):
        order = []
        for _, tsub in sub.groupby("topic"):
            idx = tsub.index.to_numpy().copy()
            rng.shuffle(idx)
            order += list(idx)
        for i, idx in enumerate(order):
            rows.append((s, i % N_GROUPS, raw.at[idx, "topic"], raw.at[idx, "soal"],
                         raw.at[idx, "jawaban"]))
    d = pd.DataFrame(rows, columns=["siswa", "chunk", "topic", "soal", "jawaban"])
    df = (d.sort_values(["siswa", "chunk", "topic", "soal"])
            .groupby(["siswa", "chunk"])["jawaban"].apply(" ".join).reset_index())
    texts, authors = _to_numpy(df["jawaban"]), _to_numpy(df["siswa"])
    skf = StratifiedKFold(N_GROUPS, shuffle=True, random_state=seed)
    folds = [(tr, te, f"fold{i+1}")
             for i, (tr, te) in enumerate(skf.split(np.zeros(len(authors)), authors))]
    return texts, authors, folds


def answer_level(raw: pd.DataFrame, seed: int):
    """Satu jawaban = satu dokumen, fold = kelompok soal yang sama."""
    gmap = question_groups(seed)
    grp = np.array([gmap[(t, int(q))] for t, q in zip(raw["topic"], raw["soal"])])
    return _to_numpy(raw["jawaban"]), _to_numpy(raw["siswa"]), group_folds(grp)
