"""SVM side of the review tasks: 15, 16, 18, 19 (C), 24 (SVM columns).

    python src_review/run_svm_review.py          # CPU, ~30-60 min

For every condition x seed x feature set (TTR / MATTR) x tuning (fixed / nested):
    stylometric only, char n-gram 2-4, char + stylometric
fixed  = C=1, weight=0.1   (what the paper reports now)
nested = C from {0.1,1,10,100}; for the hybrid also weight from
         {0.01,0.05,0.1,0.5,1.0}, chosen jointly on the inner validation fold.

Output (results_review/tables/):
    svm_results.csv        one row per run (mean +- SD over 4 folds)
    svm_perfold.csv        per fold, with the chosen C / weight and validation acc
    svm_preds.csv          per-document predictions (for McNemar)
    weight_testfold_diagnostic.csv
        test-fold accuracy for each weight at C=1 -- DIAGNOSTIC ONLY, it shows
        what a selection on test folds would have picked. Never use it to choose.
"""
import sys
import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_common as R                                     # noqa: E402

MODES = ["stylo", "char", "hybrid"]
LABEL = {"stylo": "stylometric only", "char": "char n-gram 2-4",
         "hybrid": "char + stylometric"}


def jobs():
    for s in R.SEEDS:
        yield "question groups", s
    for s in R.SEEDS:
        yield "shuffle", s
    yield "topic held out", None


def main():
    t0 = time.time()
    res, pf, pr = [], [], []
    for cname, seed in jobs():
        cond = R.CONDITIONS[cname](seed)
        cache = {}
        for tune in (False, True):
            for ttr in ("ttr", "mattr"):
                modes = MODES if ttr == "ttr" else ["stylo", "hybrid"]
                out = R.run_svm(cond, modes, ttr, tune, cache)
                for m, (summ, rows, preds) in out.items():
                    tag = dict(condition=cname, seed=seed, config=LABEL[m],
                               features=ttr if m != "char" else "-",
                               tuning="nested" if tune else "fixed")
                    res.append({**tag, **summ})
                    pf += [{**tag, **r} for r in rows]
                    pr += [{**tag, "doc": i, "fold": cond["tags"][cond["fold_of"][i]],
                            "y_true": cond["y"][i], "y_pred": preds[i]}
                           for i in range(len(preds))]
                    print(f"[{time.time()-t0:6.0f}s] {cname:<15} seed {seed!s:<4} "
                          f"{tag['tuning']:<6} {tag['features']:<5} {LABEL[m]:<20} "
                          f"acc {summ['acc_mean']:5.1f} ±{summ['acc_std']:4.1f}", flush=True)
        pd.DataFrame(res).to_csv(R.OUT / "svm_results.csv", index=False)
        pd.DataFrame(pf).to_csv(R.OUT / "svm_perfold.csv", index=False)
        pd.DataFrame(pr).to_csv(R.OUT / "svm_preds.csv", index=False)

    # --- task 18 diagnostic: where could 0.1 have come from? ----------------
    rows = []
    for cname, seed in [("topic held out", None), ("question groups", R.C.SEED)]:
        cond = R.CONDITIONS[cname](seed)
        cache = {}
        S = R.stylo_matrix(cond["texts"], "ttr")
        for w in R.W_GRID + [2.0, 5.0]:
            accs = []
            for tr, te, k in R.outer_folds(cond):
                f = R.SplitFeatures(cond["texts"], tr, te, S, char=False)
                f.Xc = R._char_cached(cache, cond["texts"], tr, te, ("outer", k))
                Xa, Xb = f.get("hybrid", w)
                accs.append((R.svm_predict(Xa, cond["y"][tr], Xb, 1.0) == cond["y"][te]).mean())
            rows.append({"condition": cname, "weight": w,
                         "test_acc_mean": np.mean(accs) * 100, "test_acc_std": np.std(accs) * 100})
            print(f"diag {cname:<15} w={w:<5} test acc {np.mean(accs)*100:5.1f}", flush=True)
    pd.DataFrame(rows).to_csv(R.OUT / "weight_testfold_diagnostic.csv", index=False)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
