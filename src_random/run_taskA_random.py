"""Atribusi penulis pada korpus chunk campuran-topik dengan split acak.

    python src_random/run_taskA_random.py

Output (results_random/tables/):
    corpus_chunks.csv          statistik dokumen chunk (seed utama)
    taskA_results.csv          4 konfigurasi, seed utama, mean +- std 4 fold
    taskA_perfold.csv          per fold
    taskA_seed_check.csv       5 pengacakan kelompok soal (semua konfigurasi)
    taskA_naive_check.csv      pembanding: pengocokan jawaban per siswa
    taskA_answer_level.csv     pembanding: satu jawaban = satu dokumen
"""
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import config as C                                   # noqa: E402
from run_taskA import evaluate                        # noqa: E402
import chunks as K                                    # noqa: E402

OUT = C.ROOT / "results_random" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [C.SEED, 1, 2, 3, 4]
CONFIGS = [
    ("length only",        "length", None),
    ("stylometric only",   "stylo",  None),
    ("char n-gram 2-4",    "char",   None),
    ("char + stylometric", "hybrid", C.STYLO_WEIGHT),
]


def main():
    raw = K.load_answers()
    texts, y, groups, df = K.build_chunks(raw, C.SEED)
    folds = K.group_folds(groups)
    chance = 100 / len(set(y))

    # --- korpus -----------------------------------------------------------
    stats = {
        "n_docs": len(df), "n_authors": df.siswa.nunique(),
        "docs_per_author": sorted(df.siswa.value_counts().unique().tolist()),
        "median_words": float(df.n_kata.median()),
        "min_words": int(df.n_kata.min()), "max_words": int(df.n_kata.max()),
        "median_answers": float(df.n_jawaban.median()),
        "docs_with_4_topics": int((df.n_topik == 4).sum()),
    }
    pd.DataFrame([stats]).to_csv(OUT / "corpus_chunks.csv", index=False)
    print("korpus chunk:", stats, f"\nchance = {chance:.1f}%\n")

    # --- hasil utama (seed utama) -------------------------------------------
    rows, pf_rows = [], []
    for label, mode, w in CONFIGS:
        res, pf = evaluate(texts, y, folds, mode, w)
        rows.append({"config": label, **res})
        pf_rows += [{"config": label, **r} for r in pf]
        print(f"  {label:<20} acc {res['acc_mean']:5.1f} ±{res['acc_std']:4.1f}"
              f"   macroF1 {res['f1_mean']:5.1f} ±{res['f1_std']:4.1f}")
    pd.DataFrame(rows).to_csv(OUT / "taskA_results.csv", index=False)
    pd.DataFrame(pf_rows).to_csv(OUT / "taskA_perfold.csv", index=False)

    # --- cek 5 pengacakan kelompok soal -------------------------------------
    print("\nCek 5 pengacakan kelompok soal")
    seed_rows = []
    for s in SEEDS:
        t_s, y_s, g_s, _ = K.build_chunks(raw, s)
        f_s = K.group_folds(g_s)
        for label, mode, w in CONFIGS:
            res, _ = evaluate(t_s, y_s, f_s, mode, w)
            seed_rows.append({"seed": s, "config": label, **res})
    sc = pd.DataFrame(seed_rows)
    sc.to_csv(OUT / "taskA_seed_check.csv", index=False)
    print(sc.groupby("config")[["acc_mean", "f1_mean"]]
            .agg(["mean", "std", "min", "max"]).round(1))

    # --- pembanding: pengocokan per siswa ------------------------------------
    print("\nPembanding: pengocokan jawaban per siswa (jebakan level soal)")
    nv_rows = []
    for s in SEEDS:
        t_n, y_n, f_n = K.naive_chunks(raw, s)
        for label, mode, w in CONFIGS[2:]:
            res, _ = evaluate(t_n, y_n, f_n, mode, w)
            nv_rows.append({"seed": s, "config": label, **res})
    nv = pd.DataFrame(nv_rows)
    nv.to_csv(OUT / "taskA_naive_check.csv", index=False)
    print(nv.groupby("config")[["acc_mean", "f1_mean"]].agg(["mean", "std"]).round(1))

    # --- pembanding: satu jawaban = satu dokumen ------------------------------
    t_a, y_a, f_a = K.answer_level(raw, C.SEED)
    res, _ = evaluate(t_a, y_a, f_a, "char")
    pd.DataFrame([{"config": "char n-gram 2-4", "n_docs": len(t_a), **res}]
                 ).to_csv(OUT / "taskA_answer_level.csv", index=False)
    print(f"\nSatu jawaban per dokumen: acc {res['acc_mean']:.1f} ±{res['acc_std']:.1f}"
          f"  macroF1 {res['f1_mean']:.1f}")


if __name__ == "__main__":
    main()
