"""Task 8: topic-feature ablation under all three conditions.

    python src_review/run_ablation_review.py      # CPU, ~10-20 min

Per split: char n-gram 2-4 vectoriser fitted on the training documents; every
training ANSWER (which carries its topic label) is transformed with it and the
features are ranked by ANOVA F against topic (f_classif, as in the earlier
run_mechanism.py). The top k% are removed, k = 0, 25, 50, 75. Test documents
and test answers are never used for the ranking. C is chosen by the same nested
rule as everywhere else, with the ablation applied inside the inner split too,
so k = 0 equals the nested char n-gram row of svm_results.csv.

Output: results_review/tables/ablation.csv, ablation_perfold.csv
"""
import sys
import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_common as R                                     # noqa: E402

LEVELS = [0.0, 0.25, 0.50, 0.75]


def ablated(cond, tr, te, frac):
    texts, ans = cond["texts"], cond["answers"]
    vec = TfidfVectorizer(**R.C.CHAR_NGRAM)
    Xtr = vec.fit_transform(texts[tr])
    Xte = vec.transform(texts[te])
    if frac > 0:
        a = ans[ans["doc"].isin(set(tr.tolist()))]
        Fv = np.nan_to_num(f_classif(vec.transform(a["jawaban"].tolist()), a["topic"].to_numpy())[0])
        n_keep = len(Fv) - int(len(Fv) * frac)
        keep = np.sort(np.argsort(Fv, kind="stable")[:n_keep])
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    return Xtr.tocsr(), Xte.tocsr()


def run(cond, frac):
    y = cond["y"]
    rows = []
    for tr, te, k in R.outer_folds(cond):
        itr, ival, _ = R.inner_split(cond, k)
        Xa, Xb = ablated(cond, itr, ival, frac)
        scores = [accuracy_score(y[ival], R.svm_predict(Xa, y[itr], Xb, c)) for c in R.C_GRID]
        c, v = R.select_best(R.C_GRID, scores)
        Xa, Xb = ablated(cond, tr, te, frac)
        p = R.svm_predict(Xa, y[tr], Xb, c)
        rows.append({"fold": cond["tags"][k], "C": c, "val_acc": v * 100,
                     "acc": accuracy_score(y[te], p) * 100,
                     "macro_f1": f1_score(y[te], p, average="macro", zero_division=0) * 100})
    return rows


def main():
    t0 = time.time()
    res, pf = [], []
    plan = [("question groups", s) for s in R.SEEDS] + \
           [("shuffle", s) for s in R.SEEDS] + [("topic held out", None)]
    for cname, seed in plan:
        cond = R.CONDITIONS[cname](seed)
        for frac in LEVELS:
            rows = run(cond, frac)
            acc = np.array([r["acc"] for r in rows])
            res.append({"condition": cname, "seed": seed, "drop_frac": frac,
                        "acc_mean": acc.mean(), "acc_std": acc.std(),
                        "f1_mean": np.mean([r["macro_f1"] for r in rows])})
            pf += [{"condition": cname, "seed": seed, "drop_frac": frac, **r} for r in rows]
            print(f"[{time.time()-t0:5.0f}s] {cname:<15} seed {seed!s:<4} drop {int(frac*100):>2}%  "
                  f"acc {acc.mean():5.1f} ±{acc.std():4.1f}", flush=True)
        pd.DataFrame(res).to_csv(R.OUT / "ablation.csv", index=False)
        pd.DataFrame(pf).to_csv(R.OUT / "ablation_perfold.csv", index=False)


if __name__ == "__main__":
    main()
