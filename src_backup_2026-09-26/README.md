# Authorship Attribution — Jawaban Ujian Esai Bahasa Indonesia

## Struktur

```
data/raw/          4 file Excel (taruh di sini)
data/processed/    hasil build_corpus.py (dibuat otomatis)
src/               semua kode
                     config.py   semua konstanta
                     corpus.py   loader (kebal backend pyarrow)
                     features.py stilometri + vectoriser
                     splits.py   skema pembagian data
model/             checkpoint IndoBERT (dibuat otomatis)
results/tables/    hasil dalam CSV
results/figures/   grafik PNG
```

## Cara pakai

```bash
pip install pandas openpyxl scikit-learn scipy matplotlib

python src/run_all.py           # langkah 1-4, jalan di CPU, ~2 menit
python src/run_indobert.py      # langkah 5, butuh GPU
```

Kalau mau satu per satu:

```bash
cd src
python build_corpus.py     # Excel -> korpus agregat
python run_taskA.py        # atribusi + baseline + dua skema split
python run_mechanism.py    # kenapa random split gagal
python run_taskB.py        # deteksi genuine vs impostor
python run_indobert.py     # terakhir, butuh GPU
```

## Angka yang harus keluar

Kalau hasilmu beda dari ini, ada yang berubah di pipeline — telusuri dulu
sebelum lanjut.

**Korpus**

| tahap | jumlah | penulis | median kata |
|---|---|---|---|
| jawaban mentah | 2.162 | 59 | 17 |
| hanya siswa lengkap 4 topik | 1.988 | 50 | 17 |
| setelah agregasi | 200 dokumen | 50 | 200 |

**Task A** (chance 2.0%)

| konfigurasi | LOTO acc | LOTO F1 | random acc | random F1 |
|---|---|---|---|---|
| length only | 3.0 ±1.0 | 1.0 ±0.9 | 3.0 ±2.2 | 1.0 ±1.1 |
| stylometric only | 5.0 ±2.2 | 4.0 ±2.0 | 6.5 ±3.0 | 4.7 ±2.1 |
| **char n-gram 2-4** | **24.0 ±5.5** | **17.6 ±4.9** | 3.5 ±2.2 | 2.8 ±1.7 |
| char + stylometric | 24.5 ±5.7 | 18.4 ±5.4 | 3.5 ±2.2 | 2.8 ±1.7 |

**ΔmacroF1 = random − LOTO = −14.8** (char n-gram). Tandanya negatif.

**Mekanisme**

- Dari fitur yang sama: prediksi topik 100.0%, prediksi penulis 3.5%
- Ablasi fitur topik: random naik 3.5 → 11.5%, LOTO turun 24.0 → 15.0%

**Task B**

AUC 0.771, 95% CI [0.736, 0.804], EER 29.6%, TPR 13.5% @ FPR 1%

## Keputusan yang jangan diubah

1. **Satu dokumen = satu siswa × satu topik.** Jawaban individual median 17 kata,
   terlalu pendek untuk stilometri. Diuji: jawaban individual sebagai dokumen
   hanya mencapai 8.1% acc.
2. **`lowercase=False`.** Lowercasing menurunkan akurasi 24.0 → 18.0%.
3. **Leave-one-topic-out sebagai skema utama.** Random split hanya pembanding.
4. **Bobot stilometri 0.1.** Tanpa pembobotan, blok stilometri mendominasi dan
   akurasi jatuh ke 7.0%.
5. **Yule's K tidak dipakai.** Butuh ribuan kata agar stabil.

## Dua hal yang gampang salah dilaporkan

**Jangan klaim hybrid menang.** 17.6% → 18.4% masih jauh di dalam std ±4.9.
Laporkan sebagai setara.

**Jangan pisahkan AUC dari titik operasinya.** AUC 0.771 terdengar bagus,
tapi TPR 13.5% di FPR 1% berarti kurang dari 1 dari 7 kasus terdeteksi pada
ambang yang aman. Keduanya harus ditulis.

## Catatan kalibrasi

`CalibratedClassifierCV` memakai CV internal, yang memakan data pada korpus
sekecil ini. Akurasi atribusi model terkalibrasi lebih rendah daripada model
tanpa kalibrasi.

Laporkan Task A dari model tanpa kalibrasi, Task B dari yang terkalibrasi.
Jangan dicampur dalam satu tabel tanpa keterangan.


## Catatan lingkungan

Kalau pyarrow terpasang, pandas membaca kolom string sebagai ArrowStringArray
dan `.values` tidak mengembalikan numpy array. scikit-learn akan melempar

    TypeError: only integer scalar arrays can be converted to a scalar index

Karena itu semua script memuat data lewat `corpus.load_docs()`, yang selalu
mengembalikan numpy array biasa. Jangan membaca CSV langsung dengan
`pd.read_csv(...)["kolom"].values` di script baru — pakai loader ini.

Sudah diuji jalan dengan dan tanpa pyarrow (pandas 3.0.2, scikit-learn 1.8.0).
