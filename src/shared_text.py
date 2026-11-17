"""Langkah 1b — jawaban kembar antar siswa (Section 3.3 di paper).

Banyak soal menanyakan fakta dengan jawaban baku (lima sila Pancasila, lima
partai bulu tangkis, kepanjangan singkatan), dan sebagian siswa menyalin dari
sumber online yang sama. Jawaban yang identik dengan jawaban siswa lain TIDAK
membawa informasi tentang penulisnya, dan pada random split ia menjadi
"jembatan" teks langsung antara dokumen uji dan dokumen latih siswa lain di
topik yang sama. Jawaban ini sengaja TIDAK dibuang (korpus dievaluasi apa
adanya); script ini hanya menghitung dan melaporkannya.

Dihitung pada korpus yang dipakai (siswa lengkap 4 topik, dari docs_agg.csv):
  - jawaban yang identik dengan jawaban >= 1 siswa lain pada soal yang sama
  - dokumen (siswa x topik) yang memuat >= 1 jawaban seperti itu
  - jawaban yang identik dengan kunci jawaban / dengan teks soalnya
  - rincian per topik
Identik = sama setelah lowercase dan buang tanda baca (spasi dinormalkan).

Output: results/tables/shared_text.csv
        results/tables/shared_text_by_topic.csv
"""
import re

import pandas as pd

import config as C
from corpus import load_raw, load_docs


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split())


def read_soal() -> pd.DataFrame:
    """Sheet 'Soal' tiap file: kolom No / Soal / Kunci Jawaban (header di baris 1)."""
    rows = []
    for topic, fname in C.TOPICS.items():
        sh = pd.read_excel(C.DATA_RAW / fname, sheet_name="Soal", header=None)
        header = [str(x).strip() for x in sh.iloc[C.SHEET_HEADER_ROW].tolist()]
        body = sh.iloc[C.SHEET_DATA_ROW:].copy()
        body.columns = header
        body = body.dropna(subset=["No"])
        for _, r in body.iterrows():
            rows.append({"topic": topic, "soal": int(r["No"]),
                         "soal_norm": norm(r["Soal"]),
                         "kunci_norm": norm(r["Kunci Jawaban"])})
    return pd.DataFrame(rows)


def main():
    _, _, _, raw = load_raw()
    _, _, _, agg = load_docs()
    raw = raw[raw["siswa"].isin(agg["siswa"].unique())].copy()   # 1988 jawaban
    raw["norm"] = raw["jawaban"].map(norm)

    # berapa siswa berbeda yang memberi jawaban (ternormalisasi) ini pada soal ini
    n_siswa = raw.groupby(["topic", "soal", "norm"])["siswa"].transform("nunique")
    raw["kembar"] = n_siswa > 1

    soal = read_soal()
    raw = raw.merge(soal, on=["topic", "soal"], how="left")
    raw["sama_kunci"] = raw["norm"] == raw["kunci_norm"]
    raw["sama_soal"] = raw["norm"] == raw["soal_norm"]

    n_ans = len(raw)
    n_kembar = int(raw["kembar"].sum())
    n_doc = raw.groupby(["siswa", "topic"]).size().shape[0]
    n_doc_kembar = int(raw.groupby(["siswa", "topic"])["kembar"].any().sum())
    n_teks_kembar = int(raw.groupby(["topic", "soal", "norm"])["siswa"]
                        .nunique().gt(1).sum())

    summary = pd.DataFrame([
        {"ukuran": "jawaban dipakai", "nilai": n_ans},
        {"ukuran": "jawaban identik dgn jawaban siswa lain (soal sama)", "nilai": n_kembar},
        {"ukuran": "persen jawaban kembar", "nilai": round(100 * n_kembar / n_ans, 1)},
        {"ukuran": "teks jawaban berbeda yang dipakai >=2 siswa", "nilai": n_teks_kembar},
        {"ukuran": "dokumen (siswa x topik)", "nilai": n_doc},
        {"ukuran": "dokumen dengan >=1 jawaban kembar", "nilai": n_doc_kembar},
        {"ukuran": "jawaban identik dgn kunci jawaban", "nilai": int(raw["sama_kunci"].sum())},
        {"ukuran": "jawaban identik dgn teks soal", "nilai": int(raw["sama_soal"].sum())},
    ])
    by_topic = (raw.groupby("topic")
                   .agg(jawaban=("kembar", "size"), kembar=("kembar", "sum"))
                   .assign(persen=lambda d: (100 * d.kembar / d.jawaban).round(1))
                   .reset_index())

    summary.to_csv(C.TABLES / "shared_text.csv", index=False)
    by_topic.to_csv(C.TABLES / "shared_text_by_topic.csv", index=False)

    print(summary.to_string(index=False))
    print()
    print(by_topic.to_string(index=False))
    print("\nJawaban kembar TIDAK dibuang - hanya dilaporkan (Section 3.3, threats).")


if __name__ == "__main__":
    main()
