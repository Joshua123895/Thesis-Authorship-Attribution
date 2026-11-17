"""Langkah 5 — IndoBERT (JALANKAN TERAKHIR).

Dua mode:

    python run_indobert.py            # FINAL: 3 seed x 20 epoch  (dipakai di Bab 4)
    python run_indobert.py --sweep    # kurva epoch 5/10/20/30/40 (sudah dijalankan)

Kenapa multi-seed:
    Tanpa seed, konfigurasi yang sama persis menghasilkan macro-F1 6.3 di satu run
    dan 9.4 di run lain. Dengan data sekecil ini, variasi antar-inisialisasi hampir
    sebesar sinyalnya. Rata-rata beberapa seed adalah angka yang layak dilaporkan.

Kenapa 20 epoch:
    Dari sweep: macro-F1 8.2 (5) -> 9.4 (10) -> 10.1 (20) -> 9.5 (30) -> 9.1 (40),
    sementara loss training turun ke 0.085. Puncak di 20, lalu overfitting.
    Anggaran ini dipilih dari skor test, yang MENGUNTUNGKAN IndoBERT - jadi kalau
    IndoBERT tetap kalah, perbandingannya konservatif.

Butuh GPU:  pip install torch --index-url https://download.pytorch.org/whl/cu124
            pip install transformers

Output:
    results/tables/indobert_final_perfold.csv
    results/tables/indobert_final_summary.csv
    results/tables/indobert_vs_char_perfold.csv
    results/tables/indobert_epoch_curve.csv      (mode --sweep)
    results/figures/indobert_epoch_curve.png     (mode --sweep)
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import config as C
from corpus import load_docs
from splits import leave_one_topic_out

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              set_seed)
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()          # sembunyikan warning classifier MISSING
except ImportError:
    raise SystemExit(
        "Butuh torch + transformers.\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
        "  pip install transformers"
    )

FINAL_SEEDS = [42, 1, 2]
FINAL_EPOCHS = 20
SWEEP_EPOCHS = [5, 10, 20, 30, 40]
CHAR_CONFIG = "char n-gram 2-4"


# --------------------------------------------------------------------------- #
def make_deterministic(seed: int):
    """Seed semua sumber keacakan: python, numpy, torch, cuda, urutan batch."""
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                             max_length=C.INDOBERT_MAXLEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        preds += model(**batch).logits.argmax(-1).cpu().tolist()
    model.train()
    return np.array(preds)


def train_and_eval(texts_tr, y_tr, texts_te, y_te, n_classes, device,
                   seed, eval_at, verbose=True):
    """Latih sampai max(eval_at), evaluasi di tiap titik eval_at."""
    make_deterministic(seed)

    tok = AutoTokenizer.from_pretrained(C.INDOBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        C.INDOBERT_MODEL, num_labels=n_classes).to(device)

    g = torch.Generator().manual_seed(seed)      # urutan batch ikut deterministik
    dl_tr = DataLoader(EssayDataset(texts_tr, y_tr, tok),
                       batch_size=C.INDOBERT_BATCH, shuffle=True, generator=g)
    dl_te = DataLoader(EssayDataset(texts_te, y_te, tok),
                       batch_size=C.INDOBERT_BATCH)

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
        mean_loss = total / len(dl_tr)

        if epoch in eval_at:
            pred = predict(model, dl_te, device)
            acc = accuracy_score(y_te, pred) * 100
            f1 = f1_score(y_te, pred, average="macro", zero_division=0) * 100
            out.append({"epoch": epoch, "loss": mean_loss, "acc": acc, "macro_f1": f1})
            if verbose:
                print(f"      epoch {epoch:>2}  loss {mean_loss:.4f}   "
                      f"acc {acc:5.1f}  macroF1 {f1:5.1f}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def load_char_perfold():
    """Ambil hasil char n-gram per fold dari run_taskA.py."""
    p = C.TABLES / "taskA_perfold.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[(df["scheme"] == "leave-one-topic-out") & (df["config"] == CHAR_CONFIG)]
    return dict(zip(df["fold"].astype(str), df["macro_f1"]))


# --------------------------------------------------------------------------- #
def run_final(texts, y, topics, n_classes, device):
    print(f"MODE FINAL: {len(FINAL_SEEDS)} seed x {FINAL_EPOCHS} epoch x 4 fold\n")

    rows = []
    for seed in FINAL_SEEDS:
        print(f"=== seed {seed} ===")
        for train, test, tag in leave_one_topic_out(topics):
            print(f"  fold: held-out {tag}")
            res = train_and_eval(texts[train], y[train], texts[test], y[test],
                                 n_classes, device, seed, [FINAL_EPOCHS])[-1]
            rows.append({"seed": seed, "fold": tag, **res})
        print()

    perfold = pd.DataFrame(rows)
    perfold.to_csv(C.TABLES / "indobert_final_perfold.csv", index=False)

    # rata-rata per seed (atas 4 fold), lalu mean +- std antar seed
    per_seed = perfold.groupby("seed")[["acc", "macro_f1"]].mean()
    fold_std = perfold.groupby("seed")[["acc", "macro_f1"]].std().mean()

    summary = pd.DataFrame([{
        "model": "IndoBERT",
        "epochs": FINAL_EPOCHS,
        "n_seeds": len(FINAL_SEEDS),
        "acc_mean": per_seed["acc"].mean(),
        "acc_std_across_seeds": per_seed["acc"].std(),
        "acc_std_across_folds": fold_std["acc"],
        "f1_mean": per_seed["macro_f1"].mean(),
        "f1_std_across_seeds": per_seed["macro_f1"].std(),
        "f1_std_across_folds": fold_std["macro_f1"],
    }])
    summary.to_csv(C.TABLES / "indobert_final_summary.csv", index=False)

    print("=" * 72)
    print("PER SEED (rata-rata 4 fold)")
    print("=" * 72)
    for seed, r in per_seed.iterrows():
        print(f"  seed {seed:<3}  acc {r.acc:5.1f}   macroF1 {r.macro_f1:5.1f}")

    s = summary.iloc[0]
    print("\n" + "=" * 72)
    print(f"IndoBERT FINAL ({FINAL_EPOCHS} epoch, {len(FINAL_SEEDS)} seed)")
    print(f"  acc     {s.acc_mean:5.1f}   (std antar seed ±{s.acc_std_across_seeds:.1f}, "
          f"antar fold ±{s.acc_std_across_folds:.1f})")
    print(f"  macroF1 {s.f1_mean:5.1f}   (std antar seed ±{s.f1_std_across_seeds:.1f}, "
          f"antar fold ±{s.f1_std_across_folds:.1f})")
    print("=" * 72)

    # perbandingan per fold melawan char n-gram
    char = load_char_perfold()
    if char is None:
        print("\n(taskA_perfold.csv tidak ditemukan - jalankan run_all.py dulu "
              "untuk perbandingan per fold)")
        return

    bert_fold = perfold.groupby("fold")["macro_f1"].mean()
    cmp_rows, wins = [], 0
    print(f"\n{'fold':<12} {'char n-gram':>12} {'IndoBERT':>10}   pemenang")
    for fold in bert_fold.index:
        c, b = char[str(fold)], bert_fold[fold]
        win = "char n-gram" if c > b else "IndoBERT"
        wins += c > b
        cmp_rows.append({"fold": fold, "char_ngram_f1": c,
                         "indobert_f1": b, "winner": win})
        print(f"{fold:<12} {c:12.1f} {b:10.1f}   {win}")
    pd.DataFrame(cmp_rows).to_csv(C.TABLES / "indobert_vs_char_perfold.csv", index=False)

    char_mean = np.mean(list(char.values()))
    print(f"\n{'rata-rata':<12} {char_mean:12.1f} {s.f1_mean:10.1f}")
    print(f"\nchar n-gram menang di {wins} dari {len(bert_fold)} fold.")

    if wins == len(bert_fold):
        print("\nKonsisten di semua fold. Ini MENJAWAB RQ3: n-gram unggul saat")
        print("kata-per-penulis rendah, sesuai prediksi Tyo dkk.")
        print("Catatan: dengan hanya 4 fold, uji tanda (sign test) paling kecil")
        print("p = 0.0625 - jadi tulis 'konsisten di keempat fold', jangan klaim p < 0.05.")
    elif s.f1_mean > char_mean:
        print("\nIndoBERT menang rata-rata. Periksa ulang kemungkinan kebocoran.")
    else:
        print(f"\nchar n-gram unggul rata-rata, tapi TIDAK di semua fold.")
        print("Tulis apa adanya: 'unggul secara rata-rata dan pada "
              f"{wins} dari {len(bert_fold)} fold'.")
        print("Jangan klaim konsisten. Sebutkan fold mana yang dimenangkan IndoBERT.")


def run_sweep(texts, y, topics, n_classes, device):
    print(f"MODE SWEEP: evaluasi pada epoch {SWEEP_EPOCHS}, seed {C.SEED}\n")

    rows = []
    for train, test, tag in leave_one_topic_out(topics):
        print(f"  fold: held-out {tag}")
        for r in train_and_eval(texts[train], y[train], texts[test], y[test],
                                n_classes, device, C.SEED, SWEEP_EPOCHS):
            rows.append({"fold": tag, **r})
        print()

    curve = pd.DataFrame(rows)
    curve.to_csv(C.TABLES / "indobert_epoch_curve.csv", index=False)
    summ = curve.groupby("epoch").agg(
        f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
        acc_mean=("acc", "mean"), loss_mean=("loss", "mean")).reset_index()

    for _, r in summ.iterrows():
        print(f"  {int(r.epoch):>2} epoch   loss {r.loss_mean:.3f}   "
              f"acc {r.acc_mean:5.1f}   macroF1 {r.f1_mean:5.1f} ±{r.f1_std:4.1f}")

    char = load_char_perfold()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(summ.epoch, summ.f1_mean, yerr=summ.f1_std,
                marker="o", capsize=3, label="IndoBERT")
    if char:
        ax.axhline(np.mean(list(char.values())), ls="--", c="tab:red",
                   label="char n-gram 2-4")
    ax.axhline(100 / n_classes, ls=":", c="grey", lw=1, label="chance")
    ax.set_xlabel("epoch"); ax.set_ylabel("macro-F1 (%)")
    ax.set_title("IndoBERT: pengaruh anggaran epoch")
    ax.legend(); fig.tight_layout()
    fig.savefig(C.FIGURES / "indobert_epoch_curve.png", dpi=200)
    print(f"\nGrafik: {C.FIGURES / 'indobert_epoch_curve.png'}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="jalankan kurva epoch, bukan run final multi-seed")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("PERINGATAN: tidak ada GPU. Ini akan sangat lambat.\n")
    print(f"device: {device}")

    texts, authors, topics, _ = load_docs()
    le = LabelEncoder()
    y = le.fit_transform(authors)
    n_classes = len(le.classes_)

    if args.sweep:
        run_sweep(texts, y, topics, n_classes, device)
    else:
        run_final(texts, y, topics, n_classes, device)


if __name__ == "__main__":
    main()
