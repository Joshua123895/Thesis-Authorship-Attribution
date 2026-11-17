# src_random — desain chunk campuran-topik + split acak

Menggantikan desain lama (satu dokumen = satu siswa x satu topik, evaluasi
leave-one-topic-out). Kode lama di `src/` dan hasilnya di `results/` TIDAK diubah.

## Desain

1. 40 soal dibagi acak ke 4 kelompok soal (10 soal, 2-3 per topik).
2. Dokumen = jawaban satu siswa untuk satu kelompok soal -> 200 dokumen,
   4 per siswa, median ~198 kata, tiap dokumen mencampur topik.
3. Fold k = kelompok soal k sebagai data uji (150 latih / 50 uji).
4. Diulang untuk 5 pengacakan kelompok soal (seed 42, 1, 2, 3, 4).

Kenapa kelompok soal, bukan mengocok jawaban per siswa: lihat `chunks.py`.
Bukti: pengocokan per siswa jatuh ke 2.4% (chance 2.0%), desain kelompok soal 28-29%.

## Cara pakai

```bash
python src_random/run_taskA_random.py      # atribusi + cek seed + pembanding, CPU ~1.5 menit
python src_random/run_taskB_random.py      # ambang keputusan, CPU ~20 detik
python src_random/run_indobert_random.py   # butuh GPU, jalankan terakhir
```

Hasil: `results_random/tables/` dan `results_random/figures/`.

## Angka yang harus keluar (seed 42)

| konfigurasi | acc | macro-F1 |
|---|---|---|
| length only | 5.0 ±1.7 | 2.1 ±0.9 |
| stylometric only | 16.0 ±2.4 | 12.0 ±2.2 |
| char n-gram 2-4 | 28.0 ±4.9 | 20.9 ±4.0 |
| char + stylometric | 29.0 ±5.2 | 22.1 ±4.5 |

5 seed, char n-gram: acc 29.4 ±2.7 (27.0-34.0).
Satu jawaban per dokumen: 10.7%. Pengocokan per siswa: 2.4%.

Ambang keputusan: AUC 0.891 [0.868, 0.913], EER 20.0%.
Siswa jujur ditandai 1% (2 dari 200) -> nama salah tertangkap 22.2%;
5% -> 56.7%; 10% -> 68.8%.

## PERHATIAN: angka ambang di paper lama terbalik

`src/run_taskB.py` menghitung ROC dengan nama BENAR sebagai kelas positif.
"TPR 13.5% @ FPR 1%" artinya: saat 1% pasangan nama-salah lolos, 13.5% pasangan
nama-benar lolos. Paper lama membacanya sebagai "13.5% pelanggar tertangkap" —
itu keliru. Dengan kerangka yang benar (positif = nama salah, tandai jika
probabilitas penulis yang diklaim rendah), angka lama seharusnya 8.2%.
`run_taskB_random.py` sudah memakai kerangka yang benar.
