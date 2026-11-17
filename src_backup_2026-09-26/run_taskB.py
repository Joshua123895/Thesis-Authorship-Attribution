"""Langkah 4 — Task B: keputusan konsistensi gaya.

Korpus hanya berisi atribusi yang benar, jadi kasus ghostwriting disimulasikan:
  genuine  = (dokumen, penulis sebenarnya)
  impostor = (dokumen, penulis lain yang diambil acak)

Skor = probabilitas terkalibrasi untuk penulis yang DIKLAIM.

Catatan: kalibrasi memakan data (CV internal pada 3 dokumen/penulis), sehingga
akurasi atribusi model terkalibrasi LEBIH RENDAH daripada model tanpa kalibrasi.
Laporkan angka Task A dari model tanpa kalibrasi, Task B dari yang terkalibrasi.
Jangan dicampur.

Output: results/tables/taskB_results.csv, taskB_operating_points.csv
        results/figures/taskB_roc.png
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve

import config as C
from corpus import load_docs
from splits import leave_one_topic_out

N_BOOTSTRAP = 2000
FPR_TARGETS = [0.01, 0.05, 0.10]


def main():
    rng = np.random.default_rng(C.SEED)   # hanya untuk bootstrap CI
    texts, y, topics, agg = load_docs()

    all_labels, all_scores, fold_rows = [], [], []

    for train, test, tag in leave_one_topic_out(topics):
        vec = TfidfVectorizer(**C.CHAR_NGRAM)
        Xtr = vec.fit_transform(texts[train])
        Xte = vec.transform(texts[test])

        clf = CalibratedClassifierCV(
            LinearSVC(C=C.SVM_C, max_iter=10000), method="sigmoid", cv=3
        ).fit(Xtr, y[train])
        proba = clf.predict_proba(Xte)
        classes = list(clf.classes_)

        # Pasangan EKSHAUSTIF: tiap dokumen uji dipasangkan dengan penulis
        # sebenarnya (genuine) DAN dengan setiap penulis lain (impostor).
        # Deterministik - tidak bergantung seed, dan memakai seluruh informasi.
        # ROC-AUC tidak sensitif terhadap rasio kelas, jadi ketidakseimbangan
        # 1:49 tidak menjadi masalah.
        labels, scores = [], []
        for i, idx in enumerate(test):
            true_author = y[idx]
            for a in classes:
                scores.append(proba[i, classes.index(a)])
                labels.append(1 if a == true_author else 0)

        auc = roc_auc_score(labels, scores)
        fold_rows.append({"fold": tag, "auc": auc, "n_pairs": len(labels)})
        all_labels += labels; all_scores += scores
        print(f"  held-out {tag:<11} AUC {auc:.3f}  ({len(labels)} pasangan)")

    labels = np.array(all_labels); scores = np.array(all_scores)
    auc = roc_auc_score(labels, scores)

    boot = []
    for _ in range(N_BOOTSTRAP):
        k = rng.integers(0, len(labels), len(labels))
        if len(set(labels[k])) < 2:
            continue
        boot.append(roc_auc_score(labels[k], scores[k]))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    fpr, tpr, thr = roc_curve(labels, scores)
    eer = fpr[np.argmin(np.abs(fpr - (1 - tpr)))]

    ops = []
    for target in FPR_TARGETS:
        i = max(np.searchsorted(fpr, target, side="right") - 1, 0)
        ops.append({"fpr_target": target, "fpr": fpr[i],
                    "tpr": tpr[i], "threshold": thr[i]})

    print(f"\nAUC gabungan {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"EER {eer*100:.1f}%")
    for o in ops:
        print(f"  FPR {o['fpr_target']*100:4.0f}%  ->  TPR {o['tpr']*100:5.1f}%"
              f"   threshold {o['threshold']:.3f}")

    pd.DataFrame(fold_rows + [{"fold": "OVERALL", "auc": auc,
                               "ci_low": lo, "ci_high": hi, "eer": eer}]
                 ).to_csv(C.TABLES / "taskB_results.csv", index=False)
    pd.DataFrame(ops).to_csv(C.TABLES / "taskB_operating_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="chance")
    for o in ops:
        ax.plot(o["fpr"], o["tpr"], "o", ms=6)
        ax.annotate(f"FPR {o['fpr_target']*100:.0f}%",
                    (o["fpr"], o["tpr"]), textcoords="offset points",
                    xytext=(8, -4), fontsize=8)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("Task B — genuine vs impostor")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(C.FIGURES / "taskB_roc.png", dpi=200)

    print(f"\nGrafik: {C.FIGURES / 'taskB_roc.png'}")
    print("\nLaporkan AUC BERSAMA titik operasinya. AUC saja adalah overclaim.")


if __name__ == "__main__":
    main()
