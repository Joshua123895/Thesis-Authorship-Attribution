"""Konfigurasi terpusat. Semua path dan konstanta ada di sini."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "model"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

for _d in (DATA_PROC, MODEL_DIR, TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# --- korpus -----------------------------------------------------------------
TOPICS = {
    "Lifestyle": "Analisis_Essay_Grading_Lifestyle.xlsx",
    "Olahraga":  "Analisis_Essay_Grading_Olahraga.xlsx",
    "Politik":   "Analisis_Essay_Grading_Politik.xlsx",
    "Teknologi": "Analisis_Essay_Grading_Teknologi.xlsx",
}
N_QUESTIONS = 10
SHEET_HEADER_ROW = 1      # baris header di sheet No.X
SHEET_DATA_ROW = 2        # data mulai di sini
COL_AUTHOR = "Siswa"
COL_TEXT = "Jawaban"

SEED = 42

# --- fitur ------------------------------------------------------------------
CHAR_NGRAM = dict(
    analyzer="char_wb",
    ngram_range=(2, 4),
    min_df=2,
    sublinear_tf=True,
    lowercase=False,          # PENTING: casing adalah sinyal gaya
)
STYLO_WEIGHT = 0.1            # bobot blok stilometri saat digabung
SVM_C = 1.0

PUNCT = list(".,;:!?-()\"'/%")

# Function word Bahasa Indonesia, disusun dari kategori gramatikal
# (pronomina, preposisi, konjungsi, partikel, auxiliary, demonstrativa).
# Sengaja BUKAN stopword list: stopword membuang kata ber-informasi rendah,
# sedangkan function word dipilih justru karena frekuensinya membedakan penulis.
FUNCTION_WORDS = """ada adalah adanya agak agar akan akhirnya aku akulah alias amat antara apa apabila
apakah apalagi atau ataupun bagai bagaikan bagi bahkan bahwa banyak beberapa begini begitu belum
berbagai berikut berkata bersama beserta betul bila bilamana bisa boleh buat bukan bukankah bukanlah
cukup dahulu dalam dan dapat dari daripada demi demikian dengan di dia dialah dini dirinya dong dst dsb
engkau entah guna hal hampir hanya hendak hingga ia ialah ibarat ini inilah itu itulah jadi jangan
jangankan janganlah jika jikalau juga justru kalau kalaupun kami kamilah kamu kamulah kan kapan karena
kecuali kedua kelak kemudian kenapa kepada kerana ketika kini kita kitalah kok kurang lagi lah lain
lainnya lalu lama lebih lewat maka makanya malah mana manakala masih masing maupun melainkan melalui
memang mengapa mereka merekalah meski meskipun mu mula mulai mungkin nah namun nanti nya oleh pada
padahal paling para pasti per pernah pula pun rupanya saat saja sama sambil sampai sana sang sangat
saya sayalah sebab sebagai sebagaimana sebaliknya sebelum sebelumnya sebenarnya sebuah secara sedang
sedangkan sedemikian segala sehingga sejak sekali sekalian sekarang sekitar selain selalu selama
seluruh semua semakin sementara sendiri seolah seperti sering serta sesuatu sesudah setelah setiap
sewaktu siapa sini situ suatu sudah supaya tadi tak tanpa tapi tentang tentu terhadap terlalu termasuk
tersebut tetapi tiap tidak tidaklah toh tsb untuk waktu walau walaupun yaitu yakni yang""".split()

# --- IndoBERT ---------------------------------------------------------------
INDOBERT_MODEL = "indobenchmark/indobert-base-p1"
INDOBERT_EPOCHS = 10
INDOBERT_LR = 2e-5
INDOBERT_BATCH = 8
INDOBERT_MAXLEN = 512
