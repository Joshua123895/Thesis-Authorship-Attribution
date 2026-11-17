"""IndoBERT side of the review tasks: 19 (nested epoch budget), 24 (third
condition) and the per-document predictions McNemar needs (22/25).

RUN ON THE LAPTOP WITH THE GPU:   run_indobert_review.bat
    or  python src_review/run_indobert_review.py ["question groups"] ["topic held out"] ["shuffle"]
Default order: question groups -> topic held out -> shuffle. Stop after the
second if time is short; the shuffle condition is the least important.

Protocol per condition and outer fold k (identical to the SVM nested rule):
    1. inner sweep: train on the two inner folds (100 docs), score the
       validation fold (k+1) mod 4 after 5/10/20/30/40 epochs (one run, seed 42)
    2. epoch budget = best validation accuracy, ties -> fewer epochs
    3. refit on all 150 training docs for that many epochs, seeds 42/1/2,
       predict the 50 test docs, save every prediction
The test fold is never looked at until step 3.

Training itself is unchanged from src/run_indobert.py (same model, AdamW,
lr 2e-5, batch 8, maxlen 512, same seeding).

Resumable: inner sweeps, per-fold results and predictions are written after
every unit, and finished units are skipped on restart.

Output (results_review/tables/):
    indobert_inner.csv      validation accuracy per epoch, per condition x fold
    indobert_perfold.csv    test acc / macro-F1 per condition x seed x fold, with epochs used
    indobert_preds.csv      per-document predictions
    indobert_summary.csv
Rough time on a GTX 1660 Ti: 5-7 h per condition.
"""
import sys
import time
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_common as R                                         # noqa: E402
from run_indobert import (EssayDataset, make_deterministic,        # noqa: E402
                          torch, AutoTokenizer, AutoModelForSequenceClassification)
from torch.utils.data import DataLoader                           # noqa: E402

C = R.C
REFIT_SEEDS = [42, 1, 2]
INNER_SEED = 42
ORDER = ["question groups", "topic held out", "shuffle"]
P_INNER = R.OUT / "indobert_inner.csv"
P_FOLD = R.OUT / "indobert_perfold.csv"
P_PRED = R.OUT / "indobert_preds.csv"

_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(C.INDOBERT_MODEL)
    return _TOK


def keep_awake():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("keep-awake on (closing the lid can still sleep the laptop)")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        out += model(**batch).logits.argmax(-1).cpu().tolist()
    model.train()
    return np.array(out)


def train(texts_tr, y_tr, texts_ev, y_ev, n_classes, device, seed, eval_at):
    """Same loop as src/run_indobert.train_and_eval, but returns predictions."""
    make_deterministic(seed)
    tok = tokenizer()
    model = AutoModelForSequenceClassification.from_pretrained(
        C.INDOBERT_MODEL, num_labels=n_classes).to(device)
    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(EssayDataset(texts_tr, y_tr, tok), batch_size=C.INDOBERT_BATCH,
                       shuffle=True, generator=g)
    dl_ev = DataLoader(EssayDataset(texts_ev, y_ev, tok), batch_size=C.INDOBERT_BATCH)
    opt = torch.optim.AdamW(model.parameters(), lr=C.INDOBERT_LR)
    model.train()
    out = []
    for epoch in range(1, max(eval_at) + 1):
        total = 0.0
        for batch in dl_tr:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch in eval_at:
            p = predict(model, dl_ev, device)
            out.append({"epoch": epoch, "loss": total / len(dl_tr), "pred": p,
                        "acc": accuracy_score(y_ev, p) * 100})
            print(f"      epoch {epoch:>2}  loss {total/len(dl_tr):.4f}  acc {out[-1]['acc']:5.1f}",
                  flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def load(path, cols):
    return pd.read_csv(path, dtype={"y_true": str, "y_pred": str}) if path.exists() \
        else pd.DataFrame(columns=cols)


def run_condition(cname, device):
    seed = None if cname == "topic held out" else C.SEED
    cond = R.CONDITIONS[cname](seed)
    texts = cond["texts"]
    le = LabelEncoder().fit(cond["y"])
    y = le.transform(cond["y"])
    n_classes = len(le.classes_)

    inner = load(P_INNER, ["condition", "fold", "epoch", "loss", "val_acc"])
    perf = load(P_FOLD, ["condition", "seed", "fold", "epochs", "loss", "acc", "macro_f1"])
    preds = load(P_PRED, ["condition", "seed", "doc", "fold", "y_true", "y_pred"])

    for tr, te, k in R.outer_folds(cond):
        tag = cond["tags"][k]
        # ---- 1-2. inner sweep -> epoch budget --------------------------------
        done_inner = inner[(inner.condition == cname) & (inner.fold == tag)]
        if len(done_inner) < len(R.EPOCH_GRID):
            itr, ival, v = R.inner_split(cond, k)
            print(f"[{cname}] fold {tag}: inner sweep (validation = {cond['tags'][v]})", flush=True)
            res = train(texts[itr], y[itr], texts[ival], y[ival], n_classes, device,
                        INNER_SEED, R.EPOCH_GRID)
            inner = inner[~((inner.condition == cname) & (inner.fold == tag))]
            inner = pd.concat([inner, pd.DataFrame(
                [{"condition": cname, "fold": tag, "epoch": r["epoch"], "loss": r["loss"],
                  "val_acc": r["acc"]} for r in res])], ignore_index=True)
            inner.to_csv(P_INNER, index=False)
            done_inner = inner[(inner.condition == cname) & (inner.fold == tag)]
        d = done_inner.sort_values("epoch")
        epochs, vacc = R.select_best([int(e) for e in d.epoch], d.val_acc.astype(float).tolist())
        print(f"[{cname}] fold {tag}: epoch budget {epochs} (validation acc {vacc:.1f})", flush=True)

        # ---- 3. refit on 150, score the test fold -----------------------------
        for s in REFIT_SEEDS:
            if ((perf.condition == cname) & (perf.seed.astype(int) == s) & (perf.fold == tag)).any():
                continue
            print(f"[{cname}] fold {tag}: refit seed {s}, {epochs} epochs", flush=True)
            r = train(texts[tr], y[tr], texts[te], y[te], n_classes, device, s, [epochs])[-1]
            perf = pd.concat([perf, pd.DataFrame([{
                "condition": cname, "seed": s, "fold": tag, "epochs": epochs, "loss": r["loss"],
                "acc": r["acc"],
                "macro_f1": f1_score(y[te], r["pred"], average="macro", zero_division=0) * 100}])],
                ignore_index=True)
            preds = pd.concat([preds, pd.DataFrame({
                "condition": cname, "seed": s, "doc": te, "fold": tag,
                "y_true": le.inverse_transform(y[te]), "y_pred": le.inverse_transform(r["pred"])})],
                ignore_index=True)
            preds.to_csv(P_PRED, index=False)       # predictions first, then the fold marker
            perf.to_csv(P_FOLD, index=False)


def summarise():
    perf = pd.read_csv(P_FOLD)
    rows = []
    for cname, q in perf.groupby("condition"):
        per_seed = q.groupby("seed")[["acc", "macro_f1"]].mean()
        rows.append({"condition": cname, "n_seed_fold": len(q),
                     "epochs_per_fold": " ".join(f"{f}:{int(e)}" for f, e in
                                                 q.groupby("fold").epochs.first().items()),
                     "acc_mean": per_seed.acc.mean(), "acc_std_seeds": per_seed.acc.std(),
                     "acc_std_folds": q.groupby("seed").acc.std(ddof=0).mean(),
                     "f1_mean": per_seed.macro_f1.mean()})
    s = pd.DataFrame(rows)
    s.to_csv(R.OUT / "indobert_summary.csv", index=False)
    print(s.round(1).to_string(index=False))


def main():
    keep_awake()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: no GPU, this will take days.")
    todo = sys.argv[1:] or ORDER
    for cname in todo:
        assert cname in R.CONDITIONS, f"unknown condition {cname!r}; use one of {ORDER}"
        t0 = time.time()
        run_condition(cname, device)
        print(f"[{cname}] finished in {(time.time()-t0)/3600:.1f} h", flush=True)
        summarise()


if __name__ == "__main__":
    main()
