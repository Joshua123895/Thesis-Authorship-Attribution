"""Shared code for the supervisor-review tasks (Henry, Nov 2026).

Nothing in src/ or src_random/ is modified; this folder only imports from them.

Three evaluation conditions, all 200 documents / 50 authors / 4 folds of 150-50:
    question groups   topics matched, questions held out   (current main design)
    shuffle           per-student shuffle, uncontrolled     (src_random naive_chunks)
    topic held out    leave-one-topic-out, topics held out  (docs_agg.csv, old design)

Nested selection (one helper for every hyperparameter): for outer test fold k,
the validation fold is (k+1) mod 4, the inner training set is the other two
folds. A value is chosen on the validation fold, then the model is refitted on
all three training folds and scored on fold k. Ties go to the smaller value
(stronger regularisation / smaller stylometric weight / fewer epochs).
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src_random"))
sys.path.insert(0, str(ROOT / "src"))
import config as C                       # noqa: E402
import features as F                     # noqa: E402
import chunks as K                       # noqa: E402
from corpus import _to_numpy             # noqa: E402

OUT = ROOT / "results_review" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [C.SEED, 1, 2, 3, 4]
C_GRID = [0.1, 1.0, 10.0, 100.0]
W_GRID = [0.01, 0.05, 0.1, 0.5, 1.0]
EPOCH_GRID = [5, 10, 20, 30, 40]
MATTR_WINDOW = 50
TTR_COL = 4                      # column of type-token ratio in F.stylometric_vector
N_FOLDS = 4


# --------------------------------------------------------------------------- #
# conditions
# --------------------------------------------------------------------------- #
_RAW = None


def answers():
    global _RAW
    if _RAW is None:
        _RAW = K.load_answers()            # 1988 answers, 50 complete students
    return _RAW.copy()


def _pack(name, seed, texts, y, fold_of, tags, ans):
    fold_of = np.asarray(fold_of, dtype=int)
    return dict(name=name, seed=seed, texts=texts, y=np.asarray(y, dtype=object),
                fold_of=fold_of, tags=tags, answers=ans)


def cond_question_groups(seed):
    raw = answers()
    gmap = K.question_groups(seed)
    raw["grp"] = [gmap[(t, int(q))] for t, q in zip(raw["topic"], raw["soal"])]
    texts, y, groups, df = K.build_chunks(raw.drop(columns="grp"), seed)
    key = {(s, g): i for i, (s, g) in enumerate(zip(df["siswa"], df["grp"]))}
    raw["doc"] = [key[(s, g)] for s, g in zip(raw["siswa"], raw["grp"])]
    return _pack("question groups", seed, texts, y, groups,
                 [f"grp{g+1}" for g in range(N_FOLDS)], raw[["doc", "topic", "jawaban"]])


def cond_shuffle(seed):
    """Same documents and folds as chunks.naive_chunks(raw, seed), plus the
    answer -> document map (needed for the topic ablation)."""
    raw = answers()
    rng = np.random.default_rng(seed)
    rows = []
    for s, sub in raw.groupby("siswa"):
        order = []
        for _, tsub in sub.groupby("topic"):
            idx = tsub.index.to_numpy().copy()
            rng.shuffle(idx)
            order += list(idx)
        for i, idx in enumerate(order):
            rows.append((s, i % N_FOLDS, raw.at[idx, "topic"], raw.at[idx, "soal"],
                         raw.at[idx, "jawaban"], idx))
    d = pd.DataFrame(rows, columns=["siswa", "chunk", "topic", "soal", "jawaban", "ans"])
    df = (d.sort_values(["siswa", "chunk", "topic", "soal"])
            .groupby(["siswa", "chunk"])["jawaban"].apply(" ".join).reset_index())
    texts, y = _to_numpy(df["jawaban"]), _to_numpy(df["siswa"])
    key = {(s, c): i for i, (s, c) in enumerate(zip(df["siswa"], df["chunk"]))}
    d["doc"] = [key[(s, c)] for s, c in zip(d["siswa"], d["chunk"])]
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    fold_of = np.zeros(len(y), dtype=int)
    for i, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_of[te] = i
    return _pack("shuffle", seed, texts, y, fold_of,
                 [f"fold{i+1}" for i in range(N_FOLDS)], d[["doc", "topic", "jawaban"]])


def cond_topic_held_out(seed=None):
    """Leave-one-topic-out on docs_agg.csv (one document = student x topic).
    Fold order = sorted topic names, same as src/splits.leave_one_topic_out."""
    df = pd.read_csv(C.DATA_PROC / "docs_agg.csv")
    texts, y = _to_numpy(df["jawaban"]), _to_numpy(df["siswa"])
    topics = sorted(df["topic"].unique())
    fold_of = np.array([topics.index(t) for t in df["topic"]])
    raw = answers()
    key = {(s, t): i for i, (s, t) in enumerate(zip(df["siswa"].astype(str), df["topic"]))}
    raw["doc"] = [key[(s, t)] for s, t in zip(raw["siswa"], raw["topic"])]
    return _pack("topic held out", seed, texts, y, fold_of, topics,
                 raw[["doc", "topic", "jawaban"]])


CONDITIONS = {"question groups": cond_question_groups,
              "shuffle": cond_shuffle,
              "topic held out": cond_topic_held_out}


def outer_folds(cond):
    f = cond["fold_of"]
    return [(np.where(f != k)[0], np.where(f == k)[0], k) for k in range(N_FOLDS)]


def inner_split(cond, k):
    """Validation fold = (k+1) mod 4; inner training = the remaining two folds."""
    f = cond["fold_of"]
    v = (k + 1) % N_FOLDS
    return np.where((f != k) & (f != v))[0], np.where(f == v)[0], v


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def _tokens(text):
    return [w.strip(".,;:!?()\"'").lower() for w in text.split()]


def mattr(text, window=MATTR_WINDOW):
    """Moving-average type-token ratio (Covington & McFall 2010)."""
    tok = [t for t in _tokens(text) if t]
    n = len(tok)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tok)) / n
    c = Counter(tok[:window])
    total = len(c)
    for i in range(window, n):
        c[tok[i]] += 1
        c[tok[i - window]] -= 1
        if c[tok[i - window]] == 0:
            del c[tok[i - window]]
        total += len(c)
    return total / (window * (n - window + 1))


def stylo_matrix(texts, ttr="ttr"):
    """F.stylometric_matrix with column 4 (TTR) optionally replaced by MATTR.
    Same 244 columns either way."""
    S = F.stylometric_matrix(texts)
    check = np.array([len(set(_tokens(t))) / max(len(t.split()), 1) for t in texts[:5]])
    assert np.allclose(S[:5, TTR_COL], check), "TTR column moved in features.py"
    if ttr == "mattr":
        S[:, TTR_COL] = [mattr(t) for t in texts]
    return S


class SplitFeatures:
    """Same transforms as features.build, but cached so one split can be
    reused for many (C, weight) values."""

    def __init__(self, texts, tr, te, S=None, char=True):
        self.Xc = None
        if char:
            vec = TfidfVectorizer(**C.CHAR_NGRAM)
            self.Xc = (vec.fit_transform(texts[tr]), sparse.csr_matrix(vec.transform(texts[te])))
            self.vec = vec
        self.Xs = None
        if S is not None:
            sc = StandardScaler().fit(S[tr])
            self.Xs = (normalize(sc.transform(S[tr])), normalize(sc.transform(S[te])))

    def get(self, mode, w=None):
        if mode == "char":
            return self.Xc
        if mode == "stylo":
            return sparse.csr_matrix(self.Xs[0]), sparse.csr_matrix(self.Xs[1])
        if mode == "hybrid":
            return (sparse.hstack([self.Xc[0], sparse.csr_matrix(self.Xs[0] * w)]).tocsr(),
                    sparse.hstack([self.Xc[1], sparse.csr_matrix(self.Xs[1] * w)]).tocsr())
        raise ValueError(mode)


def svm_predict(Xtr, ytr, Xte, c):
    return LinearSVC(C=c, max_iter=10000).fit(Xtr, ytr).predict(Xte)


def param_grid(mode, tune):
    if not tune:
        return [(C.SVM_C, C.STYLO_WEIGHT if mode == "hybrid" else None)]
    if mode == "hybrid":
        return [(c, w) for c in C_GRID for w in W_GRID]
    return [(c, None) for c in C_GRID]


def select_best(values, scores):
    """Shared selection rule: highest validation accuracy, ties -> first
    (= smallest) value in the ordered grid."""
    scores = np.asarray(scores, dtype=float)
    i = int(np.flatnonzero(scores == scores.max())[0])
    return values[i], scores[i]


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def run_svm(cond, modes, ttr="ttr", tune=True, cache=None):
    """Evaluate several modes on one condition.

    Returns {mode: (summary, perfold_rows, preds)}; preds is indexed by document.
    `cache` (dict) shares char features between calls on the same condition.
    """
    texts, y = cond["texts"], cond["y"]
    need_stylo = any(m in ("stylo", "hybrid") for m in modes)
    S = stylo_matrix(texts, ttr) if need_stylo else None
    cache = {} if cache is None else cache
    out = {}
    for mode in modes:
        grid = param_grid(mode, tune)
        preds = np.empty(len(y), dtype=object)
        rows = []
        for tr, te, k in outer_folds(cond):
            val_acc = np.nan
            if len(grid) > 1:
                itr, ival, v = inner_split(cond, k)
                fi = SplitFeatures(texts, itr, ival, S, char=False)
                fi.Xc = _char_cached(cache, texts, itr, ival, ("inner", k))
                scores = []
                for c, w in grid:
                    Xa, Xb = fi.get(mode, w)
                    scores.append(accuracy_score(y[ival], svm_predict(Xa, y[itr], Xb, c)))
                (c, w), val_acc = select_best(grid, scores)
            else:
                c, w = grid[0]
            fo = SplitFeatures(texts, tr, te, S, char=False)
            fo.Xc = _char_cached(cache, texts, tr, te, ("outer", k))
            Xa, Xb = fo.get(mode, w)
            p = svm_predict(Xa, y[tr], Xb, c)
            preds[te] = p
            rows.append({"fold": cond["tags"][k], "C": c, "weight": w,
                         "val_acc": val_acc * 100 if val_acc == val_acc else np.nan,
                         "acc": accuracy_score(y[te], p) * 100,
                         "macro_f1": f1_score(y[te], p, average="macro", zero_division=0) * 100})
        acc = np.array([r["acc"] for r in rows])
        f1 = np.array([r["macro_f1"] for r in rows])
        summ = {"acc_mean": acc.mean(), "acc_std": acc.std(),
                "f1_mean": f1.mean(), "f1_std": f1.std()}
        out[mode] = (summ, rows, preds)
    return out


def _char_cached(cache, texts, tr, te, key):
    if key not in cache:
        vec = TfidfVectorizer(**C.CHAR_NGRAM)
        cache[key] = (vec.fit_transform(texts[tr]), sparse.csr_matrix(vec.transform(texts[te])))
    return cache[key]
