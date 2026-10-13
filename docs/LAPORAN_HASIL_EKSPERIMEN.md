# 📑 Laporan Hasil Analisis & Temuan Eksperimen 2b
**Large Language Model Sebagai RAGAS Evaluator pada Domain Hukum Maritim (*SinkronAIR*)**

* **Peneliti/Penguji:** Chandra
* **Afiliasi:** Badan Riset dan Inovasi Nasional (BRIN)
* **Tanggal Pelaksanaan:** [Isi Tanggal Eksekusi]
* **Dataset yang Digunakan:** `Palayaran Fix.csv` & `Perikanan Fix.csv` (Kasus Putusan Mahkamah Agung)

---

## 1. 🎯 Ringkasan Eksekutif

[Tuliskan rangkuman 1-2 paragraf mengenai hasil pengujian perbandingan model evaluator. Sebutkan model mana yang paling konsisten, model mana yang paling longgar/ketat, dan kesesuaian evaluasi terhadap kaidah hukum].

---

## 2. 🤖 Model Evaluator yang Diuji

| Kategori | Nama Model | Backend / Inference | Versi / Parameter |
| :--- | :--- | :--- | :--- |
| **Proprietary** | GPT-4o-mini | OpenAI API | `gpt-4o-mini` |
| **Open Weights** | Gemma 3:12b | Ollama / vLLM | `gemma3:12b` |
| **Open Weights** | Gemma 4:12b / e4b | Ollama / vLLM | `gemma4:e4b` |
| **Open Weights** | Qwen 2.5 14b / 32b | Ollama / vLLM | `qwen2.5:14b` |

---

## 3. 📊 Hasil Pengujian Statistik

### A. Matriks Korelasi (Parameter 2)

#### 1. Pearson Correlation Matrix ($r$)
Mengukur hubungan linear skor antar model penilai:

| Model Evaluator | GPT-4o-mini | Gemma 3:12b | Gemma 4 | Qwen 2.5 |
| :--- | :---: | :---: | :---: | :---: |
| **GPT-4o-mini** | 1.0000 | [Hasil] | [Hasil] | [Hasil] |
| **Gemma 3:12b** | [Hasil] | 1.0000 | [Hasil] | [Hasil] |
| **Gemma 4** | [Hasil] | [Hasil] | 1.0000 | [Hasil] |
| **Qwen 2.5** | [Hasil] | [Hasil] | [Hasil] | 1.0000 |

#### 2. Spearman's Rank Correlation Matrix ($\rho$)
Mengukur konsistensi urutan peringkat kualitas jawaban:

| Model Evaluator | GPT-4o-mini | Gemma 3:12b | Gemma 4 | Qwen 2.5 |
| :--- | :---: | :---: | :---: | :---: |
| **GPT-4o-mini** | 1.0000 | [Hasil] | [Hasil] | [Hasil] |
| **Gemma 3:12b** | [Hasil] | 1.0000 | [Hasil] | [Hasil] |
| **Gemma 4** | [Hasil] | [Hasil] | 1.0000 | [Hasil] |
| **Qwen 2.5** | [Hasil] | [Hasil] | [Hasil] | 1.0000 |

---

### B. Tingkat Kesepakatan / Agreement Reliability (Parameter 3)

* **Threshold Kelulusan (Pass/Fail):** $\ge 0.7$
* **Cohen's Kappa ($\kappa$) Pairwise:**
  * GPT-4o-mini vs Gemma 3: `[Hasil]`
  * GPT-4o-mini vs Gemma 4: `[Hasil]`
  * Gemma 3 vs Gemma 4: `[Hasil]`
* **Fleiss' Kappa (Seluruh Model Evaluator):** `[Hasil]`
  * *Interpretasi:* (< 0.20: Poor, 0.21-0.40: Fair, 0.41-0.60: Moderate, 0.61-0.80: Substantial, 0.81-1.00: Almost Perfect).
* **Intraclass Correlation Coefficient (ICC(2,1)):** `[Hasil]`
  * *Interpretasi Reliabilitas Kontinu:* [Tuliskan interpretasi reliabilitas data kontinu antar penilai].

---

## 4. ⚖️ Analisis Kualitatif & Alignment Reasoning (Parameter 1)

### Temuan Pola Penalaran & Bias Model
1. **Self-Evaluation Bias / Kemurahan Nilai:**
   * [Catat apakah model yang sama dengan generator cenderung memberikan skor sempurna 1.0 tanpa dasar yang kuat].
2. **Kekakuan Format vs Substansi Hukum:**
   * [Catat apakah model tertentu menolak atribusi konteks hanya karena penulisan format nomor pasal berbeda padahal substansi ayatnya sama].
3. **Halusinasi Kritik (*Hallucinated Critique*):**
   * [Catat jika model mengkritik jawaban hukum yang sebenarnya sudah sesuai regulasi UNCLOS / UU Perikanan / UU Pelayaran].

### Studi Kasus Discrepancy Tertinggi (Purposive Sampling)

#### Kasus #1: [Query ID / Pertanyaan]
* **Pertanyaan:** "[Tuliskan pertanyaan skenario]"
* **Ground Truth:** "[Kunci jawaban]"
* **Skor & Reasoning Model A (GPT-4o-mini):**
  * Skor: `[0.xx]`
  * Justifikasi: "[Kutipan teks reasoning]"
* **Skor & Reasoning Model B (Gemma 3:12b):**
  * Skor: `[0.xx]`
  * Justifikasi: "[Kutipan teks reasoning]"
* **Analisis Yuridis:**
  * [Mengapa perbedaan ini terjadi dan siapa yang secara kaidah hukum lebih tepat].

---

## 5. 💡 Kesimpulan & Rekomendasi untuk SinkronAIR

1. **Model Evaluator Terbaik / Paling Objektif:** [Sebutkan model rekomendasi].
2. **Metrik Paling Stabil:** [Misal: Faithfulness terbukti paling stabil, Context Recall sangat sensitif terhadap ketatnya judge].
3. **Rekomendasi Parameter Evaluasi:** [Saran prompt tuning, temperature 0, context window, atau hybrid judging].
