"""Jalankan seluruh pipeline kecuali IndoBERT.

    python src/run_all.py

IndoBERT dijalankan terpisah (butuh GPU):
    python src/run_indobert.py
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    ("1. Bangun korpus",      "build_corpus.py"),
    ("2. Task A - atribusi",  "run_taskA.py"),
    ("3. Analisis mekanisme", "run_mechanism.py"),
    ("4. Task B - deteksi",   "run_taskB.py"),
]

for label, script in STEPS:
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    r = subprocess.run([sys.executable, str(SRC / script)], cwd=SRC)
    if r.returncode != 0:
        sys.exit(f"\nGAGAL di {script}")

print("\n" + "=" * 72)
print("Selesai. Hasil ada di results/tables/ dan results/figures/")
print("Berikutnya (butuh GPU): python src/run_indobert.py")
print("=" * 72)
