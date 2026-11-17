"""Langkah 3 — analisis mekanisme.

Menjawab: kenapa random split jauh lebih buruk daripada leave-one-topic-out?

Dua uji:
  1. Seberapa kuat char n-gram mengkode TOPIK dibanding PENULIS?
  2. Kalau fitur paling topik-spesifik dibuang bertahap, apa yang terjadi
     pada masing-masing skema?

Output: results/tables/mechanism_topic_vs_author.csv
        results/tables/mechanism_ablation.csv
        results/figures/mechanism_ablation.png
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import config as C
from corpus import load_docs
from splits import leave_one_topic_out, random_split

ABLATION_LEVELS = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90]


def topic_vs_author(texts, y, topics):
    X = TfidfVectorizer(**C.CHAR_NGRAM).fit_transform(texts)
    out = []
    for target, label, chance in [(topics, "topik", 100 / len(set(topics))),
                                  (y, "penulis", 100 / len(set(y)))]:
        cv = StratifiedKFold(4, shuffle=True, random_state=0)
        acc = cross_val_score(LinearSVC(C=C.SVM_C, max_iter=10000),
                              X, target, cv=cv).mean() * 100
        out.append({"target": label, "accuracy": acc, "chance": chance})
        print(f"  prediksi {label:<8} acc {acc:5.1f}%   (chance {chance:.1f}%)")
    return pd.DataFrame(out)


def ablate(texts, y, topics, folds, drop_frac):
    """Buang `drop_frac` fitur dengan F-value tertinggi terhadap TOPIK."""
    accs, f1s = [], []
    for train, test, _ in folds:
        vec = TfidfVectorizer(**C.CHAR_NGRAM)
        Xtr = vec.fit_transform(texts[train])
        Xte = vec.transform(texts[test])

        if drop_frac > 0 and len(set(topics[train])) > 1:
            Fv = np.nan_to_num(f_classif(Xtr.toarray(), topics[train])[0])
            n_drop = int(len(Fv) * drop_frac)
            keep = np.argsort(Fv)[:len(Fv) - n_drop]
            Xtr, Xte = Xtr[:, keep], Xte[:, keep]

        clf = LinearSVC(C=C.SVM_C, max_iter=10000).fit(Xtr, y[train])
        pred = clf.predict(Xte)
        accs.append(accuracy_score(y[test], pred))
        f1s.append(f1_score(y[test], pred, average="macro", zero_division=0))
    return np.mean(accs) * 100, np.mean(f1s) * 100


def main():
    texts, y, topics, agg = load_docs()

    print("1. Apakah ruang fitur didominasi topik?")
    tva = topic_vs_author(texts, y, topics)
    tva.to_csv(C.TABLES / "mechanism_topic_vs_author.csv", index=False)

    print("\n2. Ablasi fitur topik")
    schemes = {
        "leave-one-topic-out": leave_one_topic_out(topics),
        "random split": random_split(y),
    }
    rows = []
    print(f"  {'buang':>6}   {'LOTO acc':>9} {'LOTO F1':>8}   {'random acc':>11} {'random F1':>10}")
    for d in ABLATION_LEVELS:
        la, lf = ablate(texts, y, topics, schemes["leave-one-topic-out"], d)
        ra, rf = ablate(texts, y, topics, schemes["random split"], d)
        rows.append({"drop_frac": d, "loto_acc": la, "loto_f1": lf,
                     "random_acc": ra, "random_f1": rf})
        print(f"  {int(d*100):5d}%   {la:9.1f} {lf:8.1f}   {ra:11.1f} {rf:10.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES / "mechanism_ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df.drop_frac * 100, df.loto_acc, "o-", label="leave-one-topic-out")
    ax.plot(df.drop_frac * 100, df.random_acc, "s-", label="random split")
    ax.axhline(100 / len(set(y)), ls="--", c="grey", lw=1, label="chance")
    ax.set_xlabel("most topic-specific features removed (%)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Topic-feature ablation")
    ax.legend(); fig.tight_layout()
    fig.savefig(C.FIGURES / "mechanism_ablation.png", dpi=200)

    print(f"\nGrafik: {C.FIGURES / 'mechanism_ablation.png'}")
    print("\nArah berlawanan = topik adalah penyebabnya.")


if __name__ == "__main__":
    main()
