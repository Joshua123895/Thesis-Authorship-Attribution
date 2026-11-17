"""Tasks 22 / 25: McNemar over the 200 paired document predictions.

    python src_review/run_mcnemar.py
Needs: svm_preds.csv (run_svm_review.py); for the IndoBERT rows also
       indobert_preds.csv (run_indobert_review.py, GPU).

The 4 folds are pooled: every document is a test document exactly once.
b = model A right, model B wrong;  c = A wrong, B right.
statsmodels mcnemar(table, exact=(b+c < 25), correction=True).
IndoBERT: one test per seed, never averaged across seeds.

Output: results_review/tables/mcnemar.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_common as R                                     # noqa: E402


def test(y, pa, pb):
    a, b_ = (pa == y), (pb == y)
    n11, b, c, n00 = int((a & b_).sum()), int((a & ~b_).sum()), int((~a & b_).sum()), int((~a & ~b_).sum())
    exact = (b + c) < 25
    r = mcnemar([[n11, b], [c, n00]], exact=exact, correction=True)
    return dict(n=len(y), acc_a=a.mean() * 100, acc_b=b_.mean() * 100, both_right=n11,
                b_a_only=b, c_b_only=c, both_wrong=n00, exact=exact,
                statistic=float(r.statistic), p=float(r.pvalue))


def svm(pr, cond, seed, config, features, tuning):
    q = pr[(pr.condition == cond) & (pr.config == config) & (pr.features == features)
           & (pr.tuning == tuning)]
    q = q[q.seed.isna()] if seed is None else q[q.seed == seed]
    q = q.sort_values("doc")
    assert len(q) == 200, (cond, seed, config, features, tuning, len(q))
    return q.y_true.to_numpy(), q.y_pred.to_numpy()


def main():
    pr = pd.read_csv(R.OUT / "svm_preds.csv", dtype={"y_true": str, "y_pred": str})
    rows = []
    # 1. char vs char + stylometric, question groups, every grouping seed
    for seed in R.SEEDS:
        for setting, feats, tuning in [("paper now (C=1, w=0.1, TTR)", "ttr", "fixed"),
                                       ("nested C/w, MATTR", "mattr", "nested")]:
            y, pa = svm(pr, "question groups", seed, "char n-gram 2-4", "-", tuning)
            _, pb = svm(pr, "question groups", seed, "char + stylometric", feats, tuning)
            rows.append({"comparison": "char vs char+stylo", "condition": "question groups",
                         "seed": seed, "setting": setting, **test(y, pa, pb)})
    # 2. char vs IndoBERT, per IndoBERT seed, every condition that has predictions
    p = R.OUT / "indobert_preds.csv"
    if p.exists():
        bp = pd.read_csv(p, dtype={"y_true": str, "y_pred": str})
        for cond in bp.condition.unique():
            svm_seed = None if cond == "topic held out" else R.C.SEED
            for bseed in sorted(bp[bp.condition == cond].seed.unique()):
                q = bp[(bp.condition == cond) & (bp.seed == bseed)].sort_values("doc")
                if len(q) != 200:
                    print(f"skip {cond} seed {bseed}: {len(q)}/200 predictions")
                    continue
                for setting, tuning in [("char C=1", "fixed"), ("char nested C", "nested")]:
                    y, pa = svm(pr, cond, svm_seed, "char n-gram 2-4", "-", tuning)
                    assert (y == q.y_true.to_numpy()).all(), "document order differs"
                    rows.append({"comparison": "char vs IndoBERT", "condition": cond,
                                 "seed": int(bseed), "setting": setting + ", IndoBERT nested epochs",
                                 **test(y, pa, q.y_pred.to_numpy())})
    else:
        print("indobert_preds.csv not found -> IndoBERT rows skipped (run the GPU script first)")
    df = pd.DataFrame(rows)
    df.to_csv(R.OUT / "mcnemar.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.drop(columns=["n"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
