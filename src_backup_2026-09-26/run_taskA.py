"""Langkah 2 — Task A: atribusi penulis (closed-set).

Menjalankan baseline + model utama pada DUA skema pembagian data,
lalu menghitung selisihnya.

Output: results/tables/taskA_results.csv, results/tables/taskA_perfold.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

import config as C
import features as F
from corpus import load_docs
from splits import leave_one_topic_out, random_split


def evaluate(texts, y, folds, mode, weight=None, C_svm=None):
    """Jalankan satu konfigurasi pada satu skema fold."""
    C_svm = C.SVM_C if C_svm is None else C_svm
    S = F.stylometric_matrix(texts) if mode in ("stylo", "hybrid") else None
    L = F.length_matrix(texts) if mode == "length" else None

    per_fold, accs, f1s = [], [], []
    for train, test, tag in folds:
        if mode == "length":
            Xtr, Xte = F.build(texts[train], texts[test], "length",
                               L[train], L[test])
        elif mode in ("stylo", "hybrid"):
            Xtr, Xte = F.build(texts[train], texts[test], mode,
                               S[train], S[test], weight)
        else:
            Xtr, Xte = F.build(texts[train], texts[test], mode)

        clf = LinearSVC(C=C_svm, max_iter=10000).fit(Xtr, y[train])
        pred = clf.predict(Xte)

        a = accuracy_score(y[test], pred)
        f = f1_score(y[test], pred, average="macro", zero_division=0)
        accs.append(a); f1s.append(f)
        per_fold.append({"fold": tag, "acc": a * 100, "macro_f1": f * 100})

    return {
        "acc_mean": np.mean(accs) * 100, "acc_std": np.std(accs) * 100,
        "f1_mean": np.mean(f1s) * 100, "f1_std": np.std(f1s) * 100,
    }, per_fold


def main():
    texts, y, topics, agg = load_docs()

    schemes = {
        "leave-one-topic-out": leave_one_topic_out(topics),
        "random split": random_split(y),
    }
    configs = [
        ("length only",        "length", None),
        ("stylometric only",   "stylo",  None),
        ("char n-gram 2-4",    "char",   None),
        ("char + stylometric", "hybrid", C.STYLO_WEIGHT),
    ]

    rows, perfold_rows = [], []
    chance = 100 / len(set(y))
    print(f"chance level = {chance:.1f}%\n")

    for sname, folds in schemes.items():
        print(f"--- {sname} ---")
        for label, mode, w in configs:
            res, pf = evaluate(texts, y, folds, mode, w)
            rows.append({"scheme": sname, "config": label, **res})
            for r in pf:
                perfold_rows.append({"scheme": sname, "config": label, **r})
            print(f"  {label:<20} acc {res['acc_mean']:5.1f} ±{res['acc_std']:4.1f}"
                  f"   macroF1 {res['f1_mean']:5.1f} ±{res['f1_std']:4.1f}")
        print()

    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES / "taskA_results.csv", index=False)
    pd.DataFrame(perfold_rows).to_csv(C.TABLES / "taskA_perfold.csv", index=False)

    # angka inti paper
    print("=" * 70)
    print("DELTA macro-F1  (random split - leave-one-topic-out)")
    print("=" * 70)
    deltas = []
    for label, _, _ in configs:
        r = df[(df.scheme == "random split") & (df.config == label)].f1_mean.iloc[0]
        l = df[(df.scheme == "leave-one-topic-out") & (df.config == label)].f1_mean.iloc[0]
        deltas.append({"config": label, "random": r, "loto": l, "delta_f1": r - l})
        print(f"  {label:<20} {r - l:+6.1f}")
    pd.DataFrame(deltas).to_csv(C.TABLES / "taskA_delta.csv", index=False)
    print("\nTanda NEGATIF berarti random split lebih buruk. Itu temuan utama,")
    print("bukan salah hitung.")


if __name__ == "__main__":
    main()
