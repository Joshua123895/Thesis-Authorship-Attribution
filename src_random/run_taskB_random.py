"""Ambang keputusan (nama benar vs nama salah) pada korpus chunk + split acak.

    python src_random/run_taskB_random.py

Sama seperti src/run_taskB.py, dengan dua perbedaan:
  1. Fold = kelompok soal (src_random/chunks.py).
  2. Kalibrasi Platt memakai CV internal per KELOMPOK SOAL di data latih
     (3 kelompok -> 3 fold), bukan StratifiedKFold, supaya data kalibrasi
     tidak kena jebakan level soal.

Output (results_random/tables/): taskB_results.csv, taskB_operating_points.csv
        results_random/figures/taskB_roc.png
"""
import sys
import warnings
from pathlib import Path
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import config as C                       # noqa: E402
import chunks as K                       # noqa: E402

OUT = C.ROOT / "results_random" / "tables"
FIG = C.ROOT / "results_random" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 2000
FPR_TARGETS = [0.01, 0.05, 0.10]


def inner_group_cv(groups_train):
    """CV internal: tiap kelompok soal di data latih jadi satu fold kalibrasi."""
    g = np.asarray(groups_train)
    return [(np.where(g != k)[0], np.where(g == k)[0]) for k in sorted(set(g))]


def main():
    rng = np.random.default_rng(C.SEED)
    raw = K.load_answers()
    texts, y, groups, _ = K.build_chunks(raw, C.SEED)

    all_labels, all_scores, fold_rows = [], [], []
    for train, test, tag in K.group_folds(groups):
        vec = TfidfVectorizer(**C.CHAR_NGRAM)
        Xtr = vec.fit_transform(texts[train])
        Xte = vec.transform(texts[test])

        clf = CalibratedClassifierCV(
            LinearSVC(C=C.SVM_C, max_iter=10000), method="sigmoid",
            cv=inner_group_cv(groups[train])).fit(Xtr, y[train])
        proba = clf.predict_proba(Xte)
        classes = list(clf.classes_)

        labels, scores = [], []
        for i, idx in enumerate(test):
            for a in classes:
                scores.append(proba[i, classes.index(a)])
                labels.append(1 if a == y[idx] else 0)
        auc = roc_auc_score(labels, scores)
        fold_rows.append({"fold": tag, "auc": auc, "n_pairs": len(labels)})
        all_labels += labels; all_scores += scores
        print(f"  {tag:<6} AUC {auc:.3f}  ({len(labels)} pasangan)")

    labels = np.array(all_labels); scores = np.array(all_scores)
    auc = roc_auc_score(labels, scores)
    boot = []
    for _ in range(N_BOOTSTRAP):
        k = rng.integers(0, len(labels), len(labels))
        if len(set(labels[k])) < 2:
            continue
        boot.append(roc_auc_score(labels[k], scores[k]))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # Kerangka ghostwriting: POSITIF = nama salah (bukan penulisnya).
    # Dokumen ditandai kalau probabilitas penulis yang diklaim RENDAH.
    #   fpr = siswa jujur yang ikut ditandai (tuduhan palsu)
    #   tpr = pasangan nama-salah yang tertangkap
    # Catatan: run_taskB.py lama menghitung ROC dengan nama BENAR sebagai positif,
    # sehingga "TPR @ FPR 1%" di sana = nama benar yang lolos, bukan pelanggar
    # yang tertangkap. Kolom code_view_* menyimpan angka versi lama itu.
    imp = 1 - labels
    fpr, tpr, thr = roc_curve(imp, -scores)
    eer = fpr[np.argmin(np.abs(fpr - (1 - tpr)))]
    fpr_c, tpr_c, _ = roc_curve(labels, scores)
    ops = []
    for target in FPR_TARGETS:
        i = max(np.searchsorted(fpr, target, side="right") - 1, 0)
        j = max(np.searchsorted(fpr_c, target, side="right") - 1, 0)
        ops.append({"honest_flagged_target": target, "honest_flagged": fpr[i],
                    "wrong_name_flagged": tpr[i],
                    "flag_if_p_claimed_below": -thr[i],
                    "n_honest_flagged": int(round(fpr[i] * labels.sum())),
                    "code_view_wrong_name_accepted": fpr_c[j],
                    "code_view_correct_name_accepted": tpr_c[j]})

    print(f"\nAUC gabungan {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]   EER {eer*100:.1f}%")
    for o in ops:
        print(f"  siswa jujur ditandai {o['honest_flagged']*100:4.1f}% "
              f"({o['n_honest_flagged']} dari {labels.sum()})  ->  nama salah "
              f"tertangkap {o['wrong_name_flagged']*100:5.1f}%   "
              f"(tandai jika p < {o['flag_if_p_claimed_below']:.4f})")

    pd.DataFrame(fold_rows + [{"fold": "OVERALL", "auc": auc, "ci_low": lo,
                               "ci_high": hi, "eer": eer}]
                 ).to_csv(OUT / "taskB_results.csv", index=False)
    pd.DataFrame(ops).to_csv(OUT / "taskB_operating_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="chance")
    for o in ops:
        ax.plot(o["honest_flagged"], o["wrong_name_flagged"], "o", ms=6)
    ax.set_xlabel("honest students flagged")
    ax.set_ylabel("incorrectly named pairs flagged")
    ax.set_title("Decision threshold (flag = likely not the author)")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(FIG / "taskB_roc.png", dpi=200)


if __name__ == "__main__":
    main()
