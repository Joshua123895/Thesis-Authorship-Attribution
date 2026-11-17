"""Langkah 1 — bangun korpus.

Baca 4 file Excel, ambil hanya kolom Siswa + Jawaban, lalu agregasi
menjadi satu dokumen per (siswa x topik).

Output:
    data/processed/answers_raw.csv   satu jawaban = satu baris
    data/processed/docs_agg.csv      satu dokumen = satu siswa x satu topik
    results/tables/corpus_stats.csv
"""
import pandas as pd
import config as C


def read_all_answers() -> pd.DataFrame:
    rows = []
    n_read = 0          # baris yang punya isi di kolom Siswa
    n_no_text = 0       # di antaranya: tanpa teks jawaban (baris rekap di kaki sheet)
    for topic, fname in C.TOPICS.items():
        path = C.DATA_RAW / fname
        for q in range(1, C.N_QUESTIONS + 1):
            sheet = pd.read_excel(path, sheet_name=f"No.{q}", header=None)
            header = [str(x).strip() for x in sheet.iloc[C.SHEET_HEADER_ROW].tolist()]
            body = sheet.iloc[C.SHEET_DATA_ROW:].copy()
            body.columns = header
            # kolom nilai (Manual, Cos, Euc, Jac, Error *) milik penelitian
            # essay-grading aslinya -> tidak dipakai
            body = body[[C.COL_AUTHOR, C.COL_TEXT]].dropna(subset=[C.COL_AUTHOR])
            for _, r in body.iterrows():
                n_read += 1
                text = r[C.COL_TEXT]
                if pd.isna(text):
                    n_no_text += 1
                    continue
                text = str(text).strip()
                if not text or text.lower() == "nan":
                    n_no_text += 1
                    continue
                rows.append({
                    "topic": topic,
                    "soal": q,
                    "siswa": str(r[C.COL_AUTHOR]).strip(),
                    "jawaban": text,
                })
    df = pd.DataFrame(rows)
    df["n_kata"] = df["jawaban"].str.split().str.len()
    df.attrs["n_read"] = n_read
    df.attrs["n_no_text"] = n_no_text
    return df


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Satu dokumen = satu siswa x satu topik (gabungan 10 jawaban).

    Hanya siswa yang hadir di SEMUA topik yang dipakai, karena skema
    leave-one-topic-out mensyaratkan tiap penulis muncul di train dan test.
    """
    coverage = raw.groupby("siswa")["topic"].nunique()
    complete = coverage[coverage == len(C.TOPICS)].index
    sub = raw[raw["siswa"].isin(complete)]

    agg = (sub.sort_values(["siswa", "topic", "soal"])
              .groupby(["siswa", "topic"])["jawaban"]
              .apply(" ".join)
              .reset_index())
    agg["n_kata"] = agg["jawaban"].str.split().str.len()
    return agg


def main():
    raw = read_all_answers()
    agg = aggregate(raw)

    raw.to_csv(C.DATA_PROC / "answers_raw.csv", index=False)
    agg.to_csv(C.DATA_PROC / "docs_agg.csv", index=False)

    stats = pd.DataFrame([
        {"tahap": "baris dibaca dari 40 sheet", "unit": "baris",
         "jumlah": raw.attrs["n_read"], "penulis": None, "median_kata": None},
        {"tahap": "jawaban mentah (baris tanpa teks dibuang)", "unit": "jawaban",
         "jumlah": len(raw), "penulis": raw.siswa.nunique(),
         "median_kata": int(raw.n_kata.median())},
        {"tahap": "hanya siswa lengkap 4 topik", "unit": "jawaban",
         "jumlah": int(raw.siswa.isin(agg.siswa.unique()).sum()),
         "penulis": agg.siswa.nunique(),
         "median_kata": int(raw[raw.siswa.isin(agg.siswa.unique())].n_kata.median())},
        {"tahap": "setelah agregasi", "unit": "dokumen",
         "jumlah": len(agg), "penulis": agg.siswa.nunique(),
         "median_kata": int(agg.n_kata.median())},
    ])
    stats.to_csv(C.TABLES / "corpus_stats.csv", index=False)

    print(stats.to_string(index=False))
    print(f"\ndokumen per penulis : {sorted(agg.siswa.value_counts().unique())}")
    print(f"dokumen per topik   : {agg.groupby('topic').size().to_dict()}")
    print(f"\ntersimpan di {C.DATA_PROC}")


if __name__ == "__main__":
    main()
