# ⚖️ RAGAS Legal Evaluator — Eksperimen 2(b)

> **Riset Pemagangan BRIN 2026:** Membandingkan dan Mengevaluasi Varian serta Parameter Large Language Model sebagai RAGAS Evaluator pada Domain Hukum Maritim (*SinkronAIR*).

---

## 📌 Daftar Isi
1. [Latar Belakang & Tujuan](#-latar-belakang--tujuan)
2. [Struktur Repositori](#-struktur-repositori)
3. [Model LLM Target](#-model-llm-target)
4. [Langkah Instalasi & Setup Lingkungan](#-langkah-instalasi--setup-lingkungan)
5. [Panduan Eksekusi Evaluasi (Step-by-Step)](#-panduan-eksekusi-evaluasi-step-by-step)
   * [A. Menggunakan Backend OpenAI (GPT-4o-mini / GPT-4o)](#a-menggunakan-backend-openai-gpt-4o-mini--gpt-4o)
   * [B. Menggunakan Backend Ollama (Gemma 3:12b, Gemma 4, Qwen 2.5)](#b-menggunakan-backend-ollama-gemma-312b-gemma-4-qwen-25)
6. [Perekaman Log Eksekusi (Bukti Eksperimen)](#-perekaman-log-eksekusi-bukti-eksperimen)
7. [Panduan Analisis Statistik Korelasi & Agreement](#-panduan-analisis-statistik-korelasi--agreement)
8. [Penyusunan Laporan Akhir Eksperimen](#-penyusunan-laporan-akhir-eksperimen)

---

## 🎯 Latar Belakang & Tujuan

Dalam evaluasi framework Retrieval-Augmented Generation (RAG) kasus hukum maritim (*SinkronAIR*), penilaian kualitas sistem (*LLM-as-a-judge*) menggunakan framework RAGAS sangat dipengaruhi oleh model evaluator yang dipilih.

**Tujuan Eksperimen 2(b):**
1. **Ekstraksi Output Reasoning:** Memastikan evaluator RAGAS tidak hanya memberikan skor numerik ($0.0 - 1.0$), tetapi **wajib menyertakan teks alasan / justifikasi logis** di balik nilai tersebut.
2. **Alignment Reasoning:** Memeriksa kesesuaian dasar pertimbangan hukum model terhadap kaidah hukum laut dan bebas dari kritik palsu (*hallucinated critique*).
3. **Konsistensi Skor (Korelasi):** Mengukur korelasi linear (**Pearson $r$**) dan konsistensi pemeringkatan kualitas (**Spearman $\rho$**) antar-model penilai.
4. **Tingkat Kesepakatan (Agreement Score):** Mengukur konsensus kategorikal antar-penilai dengan memperhitungkan faktor kebetulan melalui **Cohen's Kappa ($\kappa$)**, **Fleiss' Kappa**, dan **Intraclass Correlation Coefficient (ICC)**.

---

## 📁 Struktur Repositori

```text
ragas-test/
├── .env.example                       # Template konfigurasi environment (API Key, host)
├── .gitignore                         # Konfigurasi file yang diabaikan Git
├── requirements.txt                   # Dependensi library Python
├── README.md                          # Panduan teknis lengkap repositori ini
│
├── configs/                           # Konfigurasi eksperimen
│
├── data/
│   ├── raw/                           # Dataset sumber skenario hukum MA (kunci jawaban rujukan)
│   │   ├── Palayaran Fix.csv
│   │   └── Perikanan Fix.csv
│   ├── qa_rag/                        # Berkas input QA siap uji (GraphRAG / MiniRAG & Naive RAG)
│   │   ├── perikanan_graphrag_minirag_qa.csv  # Output sistem GraphRAG (kolom minirag & Context graf)
│   │   ├── pelayaran_graphrag_minirag_qa.csv  # Output sistem GraphRAG
│   │   ├── perikanan_naive_qa.csv             # Output baseline Naive RAG (kolom naiverag)
│   │   └── pelayaran_naive_qa.csv             # Output baseline Naive RAG
│   ├── baseline_eval/                 # Hasil evaluasi pembanding riwayat riset BRIN
│   │   ├── perikanan_graphrag_eval_gemma4.csv
│   │   ├── perikanan_eval_gemma4.csv
│   │   ├── perikanan_eval_gemma3.csv
│   │   └── perikanan_eval_gpt4omini.csv
│   └── results/                       # Folder penyimpanan berkas output evaluasi & statistik baru
├── docs/
│   └── LAPORAN_HASIL_EKSPERIMEN.md    # Template dokumen laporan akhir hasil & temuan
│
├── logs/                              # Tempat penyimpanan file log bukti eksekusi (.log)
│
├── src/
│   ├── metrics/                       # Extractor kustom metrik RAGAS dengan penalaran eksplisit
│   │   ├── faithfulness.py            # Evaluasi kesetiaan terhadap konteks + alasan
│   │   ├── context_recall.py          # Evaluasi kelengkapan pasal + atribusi
│   │   └── answer_correctness.py      # Evaluasi akurasi jawaban (TP/FP/FN + kemiripan semantik)
│   ├── evaluators/                    # Pipeline & factory evaluator
│   │   ├── llm_factory.py             # Multi-backend wrapper (Ollama, OpenAI, vLLM)
│   │   └── runner.py                  # Batch runner dengan resume checkpoint & streaming CSV
│   └── analysis/                      # Parameter analisis evaluasi
│       ├── correlation.py             # Perhitungan Pearson r & Spearman rho
│       ├── agreement.py               # Perhitungan Cohen's Kappa, Fleiss' Kappa, dan ICC
│       └── alignment.py               # Purposive sampling kasus variansi tinggi & analisis kualitatif
│
└── scripts/
    ├── run_eval.py                    # CLI runner utama evaluasi
    ├── run_eval_ollama.sh             # Runner shell otomatis untuk Ollama
    ├── run_eval_openai.sh             # Runner shell otomatis untuk OpenAI
    └── run_analysis.py                # CLI runner analisis statistik & matriks korelasi
```

---

## 🤖 Model LLM Target

| Kategori | Model Target | Backend / Parameter CLI | Keterangan |
| :--- | :--- | :--- | :--- |
| **Proprietary / API** | **GPT-5.4** | `--backend openai --model gpt-5.4` | Frontier model OpenAI untuk evaluasi akurasi & penalaran tinggi |
| **Proprietary / API** | **GPT-4o-mini** | `--backend openai --model gpt-4o-mini` | Model efisien OpenAI untuk pembanding biaya vs kualitas |
| **Open Weights** | **Gemma 3:12b** | `--backend ollama --model gemma3:12b` | Model open-weights Google Gemma 3 (12B) via Ollama |
| **Open Weights** | **Gemma 4:12b** | `--backend ollama --model gemma4:e4b` | Evaluator ketat baseline BRIN (12B) via Ollama |
| **Open Weights** | **Qwen Series** | `--backend ollama --model qwen2.5:14b` / `32b` | Model open-weights Alibaba Qwen 2.5 (14B / 32B) |

---

## 🛠️ Langkah Instalasi & Setup Lingkungan

### 1. Buat Virtual Environment (Python 3.10+)
Disarankan menggunakan `conda` atau `venv` agar tidak mengganggu paket sistem:

```bash
# Opsi A: Menggunakan Conda (Direkomendasikan jika di cluster GPU)
conda create -n ragas-eval python=3.10 -y
conda activate ragas-eval

# Opsi B: Menggunakan venv bawaan
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instal Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Konfigurasi Environment Variable
Salin template konfigurasi `.env.example` menjadi `.env`:
```bash
cp .env.example .env
```
Buka file `.env` dan masukkan kredensial yang relevan:
```ini
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
OLLAMA_HOST=http://127.0.0.1:11434
```

---

## 🚀 Panduan Eksekusi Evaluasi (Step-by-Step)

### A. Menggunakan Backend OpenAI (GPT-4o-mini / GPT-4o)

Untuk mengevaluasi dataset `data/raw/Perikanan Fix.csv`:
```bash
python scripts/run_eval.py \
    --input "data/raw/Perikanan Fix.csv" \
    --output "data/results/perikanan_eval_gpt4omini.csv" \
    --backend openai \
    --model gpt-4o-mini \
    --embedding_backend hf
```

Atau menggunakan shell runner singkat:
```bash
bash scripts/run_eval_openai.sh "data/raw/Perikanan Fix.csv" "gpt-4o-mini" "data/results/perikanan_eval_gpt4omini.csv"
```

### B. Menggunakan Backend Ollama (Gemma 3:12b, Gemma 4, Qwen 2.5)

Pastikan Ollama server sudah berjalan:
```bash
ollama serve &
# Pastikan model sudah ditarik
ollama pull gemma3:12b
ollama pull qwen2.5:14b
```

Jalankan evaluasi:
```bash
python scripts/run_eval.py \
    --input "data/raw/Perikanan Fix.csv" \
    --output "data/results/perikanan_eval_gemma3.csv" \
    --backend ollama \
    --model gemma3:12b \
    --num_ctx 131072 \
    --embedding_backend hf
```

Atau menggunakan shell runner singkat:
```bash
bash scripts/run_eval_ollama.sh "data/raw/Perikanan Fix.csv" "gemma3:12b" "data/results/perikanan_eval_gemma3.csv"
```

> **Fitur Resume Checkpoint:** Jika koneksi atau proses terputus di tengah jalan, Anda cukup menjalankan kembali perintah yang sama. Script otomatis mendeteksi baris yang telah selesai dievaluasi dan melanjutkan dari baris berikutnya tanpa menimpa data lama.

---

## 📝 Perekaman Log Eksekusi (Bukti Eksperimen)

Untuk kebutuhan bukti pelaporan magang riset (audit trail), seluruh keluaran terminal saat menjalankan evaluasi wajib direkam ke dalam file `.log`.

Gunakan perintah `tee` sehingga output tetap terlihat di terminal sekaligus tersimpan rapi:

```bash
mkdir -p logs

# Contoh eksekusi dengan pencatatan log real-time:
python scripts/run_eval.py \
    --input "data/raw/Perikanan Fix.csv" \
    --output "data/results/perikanan_eval_gemma3.csv" \
    --backend ollama \
    --model gemma3:12b \
    2>&1 | tee logs/eval_gemma3_$(date +%Y%m%d_%H%M%S).log
```

Jika dijalankan di latar belakang (*background job* di server/cluster):
```bash
nohup python scripts/run_eval.py \
    --input "data/raw/Perikanan Fix.csv" \
    --output "data/results/perikanan_eval_gemma3.csv" \
    --backend ollama \
    --model gemma3:12b \
    > logs/eval_gemma3_nohup.log 2>&1 &
```

---

## 📊 Panduan Analisis Statistik Korelasi & Agreement

Setelah seluruh varian model selesai dievaluasi dan file CSV terkumpul di `data/results/` (atau menggunakan baseline pembanding di `data/baseline_eval/`), jalankan skrip analisis komparatif berikut:

```bash
python scripts/run_analysis.py \
    --input_files \
        data/baseline_eval/perikanan_eval_gemma4.csv \
        data/baseline_eval/perikanan_eval_gemma3.csv \
        data/baseline_eval/perikanan_eval_gpt4omini.csv \
    --model_names gemma4 gemma3 gpt4omini \
    --threshold 0.7 \
    --output_dir data/results/statistical_summary/
```

### Luaran yang Dihasilkan Otomatis:
1. **Matriks Korelasi Linear (Pearson $r$):** Disimpan di `pearson_<metric>.csv`.
2. **Matriks Konsistensi Peringkat (Spearman $\rho$):** Disimpan di `spearman_<metric>.csv`.
3. **Tingkat Kesepakatan Antar-Pasangan (Cohen's Kappa $\kappa$):** Disimpan di `cohen_kappa_<metric>.csv`.
4. **Kesepakatan Multi-Model Serentak (Fleiss' Kappa):** Ditampilkan di terminal & log.
5. **Reliabilitas Kontinu Antar-Penilai (Intraclass Correlation / ICC):** Ditampilkan di terminal & log.
6. **Sampel Kasus Discrepancy Tertinggi (Alignment Sampling):** Disimpan di `high_variance_<metric>.csv` untuk bahan analisis kualitatif.

---

## 🏆 Program Benchmarking & Rekomendasi Model Terbaik

Program khusus [`scripts/benchmark_models.py`](scripts/benchmark_models.py) digunakan untuk membandingkan model-model kandidat secara langsung terhadap **satu model acuan (Gold Standard Reference Model)**, menghitung deviasi/bias, dan **menyusun narasi deskripsi alasan ilmiah mengapa model tertentu paling direkomendasikan**:

```bash
python scripts/benchmark_models.py \
    --input_files \
        data/baseline_eval/perikanan_eval_gpt4omini.csv \
        data/baseline_eval/perikanan_eval_gemma3.csv \
        data/baseline_eval/perikanan_eval_gemma4.csv \
    --model_names gpt4omini gemma3 gemma4 \
    --reference_model gpt4omini \
    --threshold 0.7 \
    --output_dir data/results/benchmark_summary/
```

### Parameter yang Dihitung terhadap Model Acuan:
* **Mean Bias ($ME$):** Mendeteksi apakah model cenderung *over-scoring* (terlalu longgar/murah nilai) atau *under-scoring* (terlalu kaku/ketat).
* **Mean Absolute Error ($MAE$):** Rata-rata deviasi absolut terhadap acuan.
* **Korelasi Linear & Ranking ($r$ & $\rho$):** Keselarasan urutan kualitas jawaban.
* **Tingkat Kesepakatan ($\kappa$ Cohen's Kappa):** Konsensus kelulusan biner.
* **Composite Alignment Score (0–100):** Skor agregat gabungan terbobot.
* **Sintesis Narasi Otomatis:** Menghasilkan dokumen laporan Markdown lengkap di `data/results/benchmark_summary/benchmark_recommendation_report.md` yang merangkum alasan ilmiah keunggulan model terbaik.

---

## 📑 Penyusunan Laporan Akhir Eksperimen

Hasil analisis statistik dan temuan kualitatif dapat dituangkan langsung ke dalam dokumen template yang telah disediakan:

👉 **[`docs/LAPORAN_HASIL_EKSPERIMEN.md`](docs/LAPORAN_HASIL_EKSPERIMEN.md)**

Format ini memuat:
* Tabel perbandingan skor metrik antar model
* Penjelasan korelasi dan agreement reliability
* Bukti kutipan teks reasoning pada kasus-kasus divergensi/disagreement tertinggi
* Rekomendasi model evaluator yang paling objektif untuk sistem hukum maritim *SinkronAIR*.
