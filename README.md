# [DOKUMEN] RAGAS Legal Evaluator — Eksperimen 2(b)

> **Riset Pemagangan BRIN 2026:** Membandingkan dan Mengevaluasi Varian serta Parameter Large Language Model sebagai RAGAS Evaluator pada Domain Hukum Maritim (*SinkronAIR*).

---

## [DAFTAR ISI]
1. [Latar Belakang & Tujuan](#latar-belakang--tujuan)
2. [Struktur Repositori](#struktur-repositori)
3. [Daftar 5 Model LLM Target](#daftar-5-model-llm-target)
4. [Langkah Instalasi & Setup Lingkungan](#langkah-instalasi--setup-lingkungan)
5. [Sintaks Eksekusi Lengkap Tiap Model](#sintaks-eksekusi-lengkap-tiap-model)
   * [1. Model Qwen 2.5 (14B) - Ollama](#1-model-qwen-25-14b---ollama)
   * [2. Model Gemma 3 (12B) - Ollama](#2-model-gemma-3-12b---ollama)
   * [3. Model Gemma 4 (12B) - Ollama](#3-model-gemma-4-12b---ollama)
   * [4. Model GPT-4o-mini - OpenAI](#4-model-gpt-4o-mini---openai)
   * [5. Model GPT-5.4 - OpenAI](#5-model-gpt-54---openai)
6. [Fitur Auto-Logging & Tampilan Reasoning](#fitur-auto-logging--tampilan-reasoning)
7. [Panduan Analisis Statistik Korelasi & Agreement](#panduan-analisis-statistik-korelasi--agreement)
8. [Program Benchmarking & Rekomendasi Model Terbaik](#program-benchmarking--rekomendasi-model-terbaik)
9. [Penyusunan Laporan Akhir Eksperimen](#penyusunan-laporan-akhir-eksperimen)

---

## Latar Belakang & Tujuan

Dalam evaluasi framework Retrieval-Augmented Generation (RAG) kasus hukum maritim (*SinkronAIR*), penilaian kualitas sistem (*LLM-as-a-judge*) menggunakan framework RAGAS sangat dipengaruhi oleh model evaluator yang dipilih.

**Tujuan Eksperimen 2(b):**
1. **Ekstraksi Output Reasoning:** Memastikan evaluator RAGAS tidak hanya memberikan skor numerik (0.0 - 1.0), tetapi **wajib menyertakan teks alasan / justifikasi logis (`[OK|FAIL]`, `[TP|FP|FN]`)** di balik nilai tersebut.
2. **Alignment Reasoning:** Memeriksa kesesuaian dasar pertimbangan hukum model terhadap kaidah hukum laut (UU No. 45 Tahun 2009) dan bebas dari kritik palsu (*hallucinated critique*).
3. **Konsistensi Skor (Korelasi):** Mengukur korelasi linear (**Pearson $r$**) dan konsistensi pemeringkatan kualitas (**Spearman $\rho$**) antar-model penilai.
4. **Tingkat Kesepakatan (Agreement Score):** Mengukur konsensus kategorikal antar-penilai dengan memperhitungkan faktor kebetulan melalui **Cohen's Kappa ($\kappa$)**, **Fleiss' Kappa**, dan **Intraclass Correlation Coefficient (ICC)**.

---

## Struktur Repositori

```text
ragas-legal-evaluator/
├── .env.example                       # Template konfigurasi environment (API Key, host)
├── .env                               # Kredensial aktif (OpenAI API Key, Ollama Host)
├── .gitignore                         # Konfigurasi file yang diabaikan Git
├── requirements.txt                   # Dependensi library Python
├── README.md                          # Panduan teknis lengkap repositori ini
│
├── configs/                           # Konfigurasi eksperimen
│
├── data/
│   ├── raw/                           # Dataset sumber skenario hukum MA
│   ├── qa_rag/                        # Berkas input QA siap uji (100 baris kasus perikanan & pelayaran)
│   │   ├── perikanan_naive_qa.csv     # Input utama evaluasi Naive RAG (100 baris)
│   │   ├── pelayaran_naive_qa.csv     # Input Naive RAG pelayaran
│   │   ├── perikanan_graphrag_minirag_qa.csv
│   │   └── pelayaran_graphrag_minirag_qa.csv
│   ├── baseline_eval_backup/          # Arsip baseline evaluasi terdahulu BRIN
│   └── results/                       # Folder output hasil evaluasi CSV, metadata JSON, & statistik
│
├── docs/
│   └── LAPORAN_HASIL_EKSPERIMEN.md    # Template dokumen laporan akhir hasil & temuan
│
├── logs/                              # Berkas log audit bukti eksekusi (.log) per run
│
├── src/
│   ├── metrics/                       # Extractor kustom metrik RAGAS dengan penalaran eksplisit
│   │   ├── faithfulness.py            # Evaluasi kesetiaan terhadap konteks + alasan [OK|FAIL]
│   │   ├── context_recall.py          # Evaluasi kelengkapan pasal + atribusi [OK|FAIL]
│   │   └── answer_correctness.py      # Evaluasi akurasi jawaban (TP/FP/FN + kemiripan semantik)
│   ├── evaluators/                    # Pipeline & factory evaluator
│   │   ├── llm_factory.py             # Multi-backend wrapper (Ollama, OpenAI, vLLM)
│   │   └── runner.py                  # Batch runner dengan resume checkpoint & streaming CSV
│   ├── analysis/                      # Parameter analisis evaluasi
│   │   ├── correlation.py             # Perhitungan Pearson r & Spearman rho
│   │   ├── agreement.py               # Perhitungan Cohen's Kappa, Fleiss' Kappa, dan ICC
│   │   ├── alignment.py               # Purposive sampling kasus variansi tinggi & analisis kualitatif
│   │   └── benchmarking.py            # Perhitungan bias, MAE, composite score, & narasi rekomendasi
│   └── utils/
│       └── logger.py                  # Logger ANSI berwarna untuk terminal & plain-text untuk file
│
└── scripts/
    ├── run_eval.py                    # CLI runner utama evaluasi
    ├── run_analysis.py                # CLI runner analisis statistik & matriks korelasi
    └── benchmark_models.py            # CLI runner benchmarking komparatif terhadap model acuan
```

---

## Daftar 5 Model LLM Target

| No | Model Target | Backend | Parameter CLI | Peran / Keterangan |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **Qwen 2.5 (14B)** | Ollama | `--backend ollama --model qwen2.5:14b` | Open-weights performa tinggi untuk penalaran domain khusus |
| 2 | **Gemma 3 (12B)** | Ollama | `--backend ollama --model gemma3:12b` | Open-weights Google generasi terbaru (12B) |
| 3 | **Gemma 4 (12B)** | Ollama | `--backend ollama --model gemma4:e4b` | Baseline evaluator ketat BRIN terdahulu |
| 4 | **GPT-4o-mini** | OpenAI | `--backend openai --model gpt-4o-mini` | Model efisien OpenAI untuk pembanding *cost-to-performance* |
| 5 | **GPT-5.4** | OpenAI | `--backend openai --model gpt-5.4` | Frontier model OpenAI sebagai *Gold Standard Reference* |

---

## Langkah Instalasi & Setup Lingkungan

### 1. Lingkungan Python
Repositori ini menggunakan Python 3.12+ dengan virtual environment di `.venv`:
```bash
# Aktifkan virtual environment yang sudah terkonfigurasi
source .venv/bin/activate
```

### 2. Service Ollama (Lokal GPU)
Pastikan server Ollama aktif pada host `127.0.0.1:11434` dan model-model lokal sudah ditarik:
```bash
ollama serve &

# Tarik model yang dibutuhkan (jika belum ada)
ollama pull qwen2.5:14b
ollama pull gemma3:12b
ollama pull gemma4:e4b
```

### 3. Konfigurasi Kredensial (.env)
Pastikan file `.env` di direktori utama telah berisi kredensial yang valid:
```ini
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
OLLAMA_HOST=http://127.0.0.1:11434
```

---

## Sintaks Eksekusi Lengkap Tiap Model

Dataset input standar untuk Tugas 2(b) adalah **`data/qa_rag/perikanan_naive_qa.csv`** (100 baris pertanyaan dan jawaban hukum maritim).

### 1. Model Qwen 2.5 (14B) - Ollama
```bash
.venv/bin/python scripts/run_eval.py \
    --input "data/qa_rag/perikanan_naive_qa.csv" \
    --output "data/results/perikanan_eval_qwen25_14b.csv" \
    --backend ollama \
    --model "qwen2.5:14b"
```

### 2. Model Gemma 3 (12B) - Ollama
```bash
.venv/bin/python scripts/run_eval.py \
    --input "data/qa_rag/perikanan_naive_qa.csv" \
    --output "data/results/perikanan_eval_gemma3.csv" \
    --backend ollama \
    --model "gemma3:12b"
```

### 3. Model Gemma 4 (12B) - Ollama
```bash
.venv/bin/python scripts/run_eval.py \
    --input "data/qa_rag/perikanan_naive_qa.csv" \
    --output "data/results/perikanan_eval_gemma4.csv" \
    --backend ollama \
    --model "gemma4:e4b"
```

### 4. Model GPT-4o-mini - OpenAI
```bash
.venv/bin/python scripts/run_eval.py \
    --input "data/qa_rag/perikanan_naive_qa.csv" \
    --output "data/results/perikanan_eval_gpt4omini.csv" \
    --backend openai \
    --model "gpt-4o-mini"
```

### 5. Model GPT-5.4 - OpenAI
```bash
.venv/bin/python scripts/run_eval.py \
    --input "data/qa_rag/perikanan_naive_qa.csv" \
    --output "data/results/perikanan_eval_gpt54.csv" \
    --backend openai \
    --model "gpt-5.4"
```

> **Fitur Resume Checkpoint:** Jika proses terhenti di tengah jalan (misal timeout atau interupsi), jalankan kembali perintah yang sama. Sistem secara otomatis mendeteksi baris yang sudah tersimpan di file output CSV dan melanjutkan sisa baris yang belum dievaluasi.

---

## Fitur Auto-Logging & Tampilan Reasoning

Setiap eksekusi `run_eval.py` otomatis mencatat dua luaran log:
1. **Terminal Console:** Menampilkan log berwarna ANSI secara real-time yang menyajikan skor numerik beserta teks justifikasi penalaran (`[OK]`, `[FAIL]`, `[TP]`, `[FP]`, `[FN]`) tanpa emoticon.
2. **Berkas Audit Trail:** Otomatis disimpan ke folder `logs/eval_<model>_<timestamp>.log` dalam bentuk teks polos bersih tanpa kode warna, siap digunakan sebagai bukti eksperimen riset.
3. **Metadata JSON Bukti Run:** Disimpan di samping file CSV hasil (`<output_csv>.meta.json`) yang memuat rekap durasi, jumlah baris, parameter run, dan skor rata-rata.

---

## Panduan Analisis Statistik Korelasi & Agreement

Setelah kelima model selesai dievaluasi dan file CSV terkumpul di folder `data/results/`, jalankan skrip analisis komparatif berikut:

```bash
.venv/bin/python scripts/run_analysis.py \
    --input_files \
        data/results/perikanan_eval_gpt54.csv \
        data/results/perikanan_eval_gpt4omini.csv \
        data/results/perikanan_eval_qwen25_14b.csv \
        data/results/perikanan_eval_gemma3.csv \
        data/results/perikanan_eval_gemma4.csv \
    --model_names gpt54 gpt4omini qwen25 gemma3 gemma4 \
    --threshold 0.7 \
    --output_dir data/results/statistical_summary/
```

### Luaran Analisis:
1. **Matriks Korelasi Linear (Pearson $r$):** Disimpan di `pearson_<metric>.csv`.
2. **Matriks Konsistensi Peringkat (Spearman $\rho$):** Disimpan di `spearman_<metric>.csv`.
3. **Tingkat Kesepakatan Antar-Pasangan (Cohen's Kappa $\kappa$):** Disimpan di `cohen_kappa_<metric>.csv`.
4. **Kesepakatan Multi-Model Serentak (Fleiss' Kappa):** Mengukur reliabilitas 5 rater sekaligus.
5. **Reliabilitas Kontinu (Intraclass Correlation / ICC):** Dihitung menggunakan model ICC(2,1).
6. **Discrepancy Reasoning Audit (Alignment Sampling):** Disimpan di `high_variance_<metric>.csv` untuk membandingkan secara langsung teks reasoning pada kasus-kasus dengan deviasi penilaian terbesar antar-model.

---

## Program Benchmarking & Rekomendasi Model Terbaik

Untuk membandingkan performa model-model kandidat terhadap satu model acuan (*Gold Standard Reference*, misal **GPT-5.4**), gunakan script `benchmark_models.py`:

```bash
.venv/bin/python scripts/benchmark_models.py \
    --input_files \
        data/results/perikanan_eval_gpt54.csv \
        data/results/perikanan_eval_gpt4omini.csv \
        data/results/perikanan_eval_qwen25_14b.csv \
        data/results/perikanan_eval_gemma3.csv \
        data/results/perikanan_eval_gemma4.csv \
    --model_names gpt54 gpt4omini qwen25 gemma3 gemma4 \
    --reference_model gpt54 \
    --threshold 0.7 \
    --output_dir data/results/benchmark_summary/
```

### Parameter yang Dihitung:
* **Mean Bias ($ME$):** Mendeteksi kecenderungan *over-scoring* (terlalu longgar) atau *under-scoring* (terlalu kaku/ketat).
* **Mean Absolute Error ($MAE$):** Rata-rata selisih absolut terhadap model acuan.
* **Korelasi Linear & Ranking ($r$ & $\rho$):** Keselarasan urutan kualitas jawaban.
* **Cohen's Kappa ($\kappa$):** Konsensus biner kelulusan pada ambang batas $\ge 0.7$.
* **Composite Alignment Score (0–100):** Skor gabungan terbobot objektif.
* **Laporan Markdown Otomatis:** Disimpan di `data/results/benchmark_summary/benchmark_recommendation_report.md` lengkap dengan narasi keunggulan model terbaik.

---

## Penyusunan Laporan Akhir Eksperimen

Hasil analisis statistik, tabel matriks, dan kutipan teks penalaran (*reasoning*) dapat dituangkan langsung ke template laporan riset:

[LAPORAN HASIL EKSPERIMEN](docs/LAPORAN_HASIL_EKSPERIMEN.md)
