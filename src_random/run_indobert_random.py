"""IndoBERT pada korpus chunk + split acak (JALANKAN DI LAPTOP DENGAN GPU).

    python src_random/run_indobert_random.py

Fold, dokumen, dan seed pengacakan kelompok soal SAMA dengan
src_random/run_taskA_random.py. Training memakai fungsi yang sama dengan
src/run_indobert.py (AdamW, lr 2e-5, batch 8, maxlen 512).

Anggaran epoch DIKUNCI 20 (dari sweep lama), tidak dipilih ulang dari skor
test -> tidak ada keuntungan tuning di data uji.

Output (results_random/tables/):
    indobert_perfold.csv        3 seed x 4 fold
    indobert_summary.csv        mean +- std
    indobert_vs_char_perfold.csv
Butuh: results_random/tables/taskA_perfold.csv (jalankan run_taskA_random.py dulu)

Bisa dilanjutkan: tiap fold disimpan langsung ke indobert_perfold.csv, dan
pasangan (seed, fold) yang sudah ada dilewati. Selama jalan, Windows diminta
tidak sleep (SetThreadExecutionState); menutup lid laptop tetap bisa membuatnya sleep.
"""
import sys
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
import config as C                                      # noqa: E402
import chunks as K                                      # noqa: E402
from run_indobert import train_and_eval, torch          # noqa: E402

SEEDS = [42, 1, 2]
EPOCHS = 20
OUT = C.ROOT / "results_random" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def keep_awake():
    """Minta Windows tidak sleep selama proses ini hidup (tidak mengubah setting)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("keep-awake aktif")


def main():
    keep_awake()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("PERINGATAN: tidak ada GPU, ini akan sangat lambat.\n")

    raw = K.load_answers()
    texts, authors, groups, _ = K.build_chunks(raw, C.SEED)
    le = LabelEncoder()
    y = le.fit_transform(authors)
    n_classes = len(le.classes_)
    folds = K.group_folds(groups)

    pf_path = OUT / "indobert_perfold.csv"
    done = pd.read_csv(pf_path) if pf_path.exists() else pd.DataFrame(
        columns=["seed", "fold", "epoch", "loss", "acc", "macro_f1"])
    have = set(zip(done["seed"].astype(int), done["fold"].astype(str)))
    if have:
        print(f"lanjut: {len(have)} fold sudah selesai, dilewati")
    for seed in SEEDS:
        print(f"=== seed {seed} ===")
        for train, test, tag in folds:
            if (seed, tag) in have:
                print(f"  fold {tag}  (sudah ada, lewati)")
                continue
            print(f"  fold {tag}")
            r = train_and_eval(texts[train], y[train], texts[test], y[test],
                               n_classes, device, seed, [EPOCHS])[-1]
            done = pd.concat([done, pd.DataFrame([{"seed": seed, "fold": tag, **r}])],
                             ignore_index=True)
            done.to_csv(pf_path, index=False)          # simpan tiap fold
    pf = done[done["seed"].isin(SEEDS)].copy()
    pf["seed"] = pf["seed"].astype(int)

    per_seed = pf.groupby("seed")[["acc", "macro_f1"]].mean()
    fold_std = pf.groupby("seed")[["acc", "macro_f1"]].std().mean()
    summ = pd.DataFrame([{
        "epochs": EPOCHS, "n_seeds": len(SEEDS),
        "acc_mean": per_seed.acc.mean(), "acc_std_seeds": per_seed.acc.std(),
        "acc_std_folds": fold_std.acc,
        "f1_mean": per_seed.macro_f1.mean(), "f1_std_seeds": per_seed.macro_f1.std(),
        "f1_std_folds": fold_std.macro_f1,
    }])
    summ.to_csv(OUT / "indobert_summary.csv", index=False)
    s = summ.iloc[0]
    print(f"\nIndoBERT acc {s.acc_mean:.1f} (seed ±{s.acc_std_seeds:.1f}, fold ±{s.acc_std_folds:.1f})"
          f"   macroF1 {s.f1_mean:.1f} (seed ±{s.f1_std_seeds:.1f}, fold ±{s.f1_std_folds:.1f})")

    pc = OUT / "taskA_perfold.csv"
    if pc.exists():
        ch = pd.read_csv(pc)
        ch = ch[ch.config == "char n-gram 2-4"].set_index("fold")["macro_f1"]
        bert = pf.groupby("fold")["macro_f1"].mean()
        cmp = pd.DataFrame({"char_ngram_f1": ch, "indobert_f1": bert})
        cmp["winner"] = np.where(cmp.char_ngram_f1 > cmp.indobert_f1,
                                 "char n-gram", "IndoBERT")
        cmp.to_csv(OUT / "indobert_vs_char_perfold.csv")
        print("\n", cmp.round(1).to_string())
        print(f"\nchar n-gram menang di {(cmp.winner == 'char n-gram').sum()} dari {len(cmp)} fold")


if __name__ == "__main__":
    main()
