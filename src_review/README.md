# src_review — tugas kode dari review Pak Henry (Nov 2026)

Tidak ada file di `src/` atau `src_random/` yang diubah. Folder ini hanya mengimpor dari sana.
Hasil: `results_review/tables/`.

| script | tugas | jalan di | waktu |
|---|---|---|---|
| `run_svm_review.py` | 15, 16, 18, 19 (C), 24 (kolom SVM) | CPU | ~15 menit |
| `run_extra_review.py` | baris length-only & single-answer di Tabel I | CPU | ~2 menit |
| `run_ablation_review.py` | 8 | CPU | ~6 menit |
| `run_indobert_review.py` (via `run_indobert_review.bat`) | 19 (epoch), 24 (kolom IndoBERT), prediksi untuk McNemar | GPU | ~5-7 jam per kondisi |
| `run_mcnemar.py` | 22, 25 | CPU (butuh `statsmodels`) | detik |

## Desain bersama (`review_common.py`)

Tiga kondisi, semuanya 200 dokumen, 50 penulis, fold 150/50:
- `question groups` — desain utama (src_random)
- `shuffle` — pengocokan per siswa (identik dengan `chunks.naive_chunks`, dicek)
- `topic held out` — leave-one-topic-out pada `docs_agg.csv`

Nested selection, satu aturan untuk semua hyperparameter: saat fold k jadi test,
fold (k+1) mod 4 jadi validasi, dua fold sisanya latih-dalam. Nilai terbaik di
validasi dipilih (seri -> nilai lebih kecil), lalu refit di tiga fold latih.
Grid: C {0.1, 1, 10, 100}; bobot stilometri {0.01, 0.05, 0.1, 0.5, 1}
(bersama C); epoch IndoBERT {5, 10, 20, 30, 40} (sweep pakai seed 42 saja,
refit seed 42/1/2).

MATTR (window 50) menggantikan kolom TTR satu-satu -> jumlah fitur tetap 244.

## Cek reproduksi (setting lama, C=1, w=0.1, TTR)

question groups seed 42: stylo 16.0, char 28.0, hybrid 29.0 — sama dengan src_random.
shuffle seed 42 char 3.0; topic held out stylo 5.0, char 24.0 — sama dengan hasil lama.
