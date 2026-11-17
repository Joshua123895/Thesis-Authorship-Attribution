"""Remaining Table I rows under nested selection: length only (5 groupings)
and single answers as documents (main grouping).

    python src_review/run_extra_review.py
Output: results_review/tables/extra_results.csv
"""
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_common as R                                     # noqa: E402


def length_nested(cond):
    L = R.F.length_matrix(cond["texts"])
    y = cond["y"]

    def fit(tr, te, c):
        sc = StandardScaler().fit(L[tr])
        return R.svm_predict(sparse.csr_matrix(sc.transform(L[tr])), y[tr],
                             sparse.csr_matrix(sc.transform(L[te])), c)
    accs, f1s = [], []
    for tr, te, k in R.outer_folds(cond):
        itr, ival, _ = R.inner_split(cond, k)
        c, _ = R.select_best(R.C_GRID, [accuracy_score(y[ival], fit(itr, ival, c)) for c in R.C_GRID])
        p = fit(tr, te, c)
        accs.append(accuracy_score(y[te], p) * 100)
        f1s.append(f1_score(y[te], p, average="macro", zero_division=0) * 100)
    return dict(acc_mean=np.mean(accs), acc_std=np.std(accs), f1_mean=np.mean(f1s), f1_std=np.std(f1s))


def single_answer_cond(seed):
    raw = R.answers()
    gmap = R.K.question_groups(seed)
    grp = np.array([gmap[(t, int(q))] for t, q in zip(raw["topic"], raw["soal"])])
    return dict(name="single answers", seed=seed, texts=R._to_numpy(raw["jawaban"]),
                y=R._to_numpy(raw["siswa"]), fold_of=grp,
                tags=[f"grp{g+1}" for g in range(4)], answers=None)


def main():
    rows = []
    for s in R.SEEDS:
        r = length_nested(R.cond_question_groups(s))
        rows.append({"run": "length only", "seed": s, **r})
        print("length", s, round(r["acc_mean"], 1), round(r["acc_std"], 1), flush=True)
    for tune in (False, True):
        out = R.run_svm(single_answer_cond(R.C.SEED), ["char"], tune=tune)
        s = out["char"][0]
        rows.append({"run": f"single answers ({'nested' if tune else 'C=1'})", "seed": R.C.SEED, **s})
        print("single", tune, round(s["acc_mean"], 1), round(s["acc_std"], 1),
              [r["C"] for r in out["char"][1]], flush=True)
    pd.DataFrame(rows).to_csv(R.OUT / "extra_results.csv", index=False)


if __name__ == "__main__":
    main()
