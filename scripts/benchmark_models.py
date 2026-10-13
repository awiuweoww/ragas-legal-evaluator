#!/usr/bin/env python3
"""
Copyright (c) 2026 Badan Riset dan Inovasi Nasional (BRIN)
THIS SOFTWARE SOURCE CODE AND ANY EXECUTABLE DERIVED THEREOF ARE PROPRIETARY
TO BRIN, AS APPLICABLE, AND SHALL NOT BE USED IN ANY WAY
OTHER THAN BEFOREHAND AGREED ON BY BRIN, NOR BE REPRODUCED
OR DISCLOSED TO THIRD PARTIES WITHOUT PRIOR WRITTEN AUTHORIZATION BY
BRIN, AS APPLICABLE.

Author      : Chandra
Version     : 0.1.0
Created     : 13-10-2026
Description : CLI script untuk membandingkan performa beberapa varian evaluator LLM
              terhadap satu model referensi acuan (Gold Standard) serta menyintesis
              deskripsi narasi alasan mengapa suatu model lebih unggul.

Changelog:
- 0.1.0 (13-10-2026): Implementasi benchmarking multi-model dan generator laporan markdown otomatis.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from src.analysis.benchmarking import evaluate_against_reference, rank_models_and_generate_reasons


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    rename_dict = {}
    for col in df.columns:
        c_low = col.lower()
        if "faithfulness" in c_low and "score" in c_low:
            rename_dict[col] = "faithfulness_score"
        elif "faithfulness" in c_low and ("explanation" in c_low or "reason" in c_low):
            rename_dict[col] = "faithfulness_reasoning"
        elif "context_recall" in c_low and "score" in c_low:
            rename_dict[col] = "context_recall_score"
        elif "context_recall" in c_low and ("explanation" in c_low or "reason" in c_low):
            rename_dict[col] = "context_recall_reasoning"
        elif "answer_correctness" in c_low and "score" in c_low:
            rename_dict[col] = "answer_correctness_score"
        elif "answer_correctness" in c_low and ("explanation" in c_low or "reason" in c_low):
            rename_dict[col] = "answer_correctness_reasoning"
        elif c_low in ["question", "query", "pertanyaan"]:
            rename_dict[col] = "query_text"
        elif c_low in ["ground_truth", "gold_answer", "reference"]:
            rename_dict[col] = "ground_truth"
        elif c_low in ["id", "query_id"]:
            rename_dict[col] = "query_id"
    return df.rename(columns=rename_dict)


def main():
    parser = argparse.ArgumentParser(description="Benchmarking Model Evaluator Terhadap Model Referensi Acuan")
    parser.add_argument("--input_files", nargs="+", required=True, help="Daftar berkas CSV hasil evaluasi per model")
    parser.add_argument("--model_names", nargs="+", required=True, help="Nama identitas masing-masing model")
    parser.add_argument("--reference_model", required=True, help="Nama model yang dijadikan referensi acuan (misal: gpt4omini)")
    parser.add_argument("--output_dir", default="data/results/benchmark_summary", help="Direktori penyimpanan laporan")
    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold biner untuk Cohen's Kappa")
    args = parser.parse_args()

    if len(args.input_files) != len(args.model_names):
        raise ValueError("Jumlah input_files harus sama persis dengan jumlah model_names!")

    if args.reference_model not in args.model_names:
        raise ValueError(
            f"Model referensi '{args.reference_model}' harus terdaftar dalam daftar --model_names {args.model_names}!"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    bench_log_path = os.path.join(logs_dir, f"benchmark_{args.reference_model}_ref.log")
    log_fp = open(bench_log_path, "w", encoding="utf-8")

    def log_print(msg=""):
        print(msg)
        log_fp.write(str(msg) + "\n")
        log_fp.flush()

    log_print("==================================================================")
    log_print(f"🎯 BENCHMARKING MULTI-MODEL RAGAS EVALUATOR (TUGAS 2b)")
    log_print(f"Model Referensi Acuan : {args.reference_model} (Gold Standard / Anchor)")
    log_print(f"Model Kandidat Diuji  : {[m for m in args.model_names if m != args.reference_model]}")
    log_print(f"Threshold Kappa       : >= {args.threshold}")
    log_print(f"Berkas Log Bukti      : {bench_log_path}")
    log_print("==================================================================\n")

    # Muat seluruh dataset evaluasi
    dfs = []
    metrics = ["faithfulness_score", "context_recall_score", "answer_correctness_score"]

    for fpath, mname in zip(args.input_files, args.model_names):
        raw_df = pd.read_csv(fpath)
        norm_df = normalize_column_names(raw_df)
        if "query_id" not in norm_df.columns:
            norm_df["query_id"] = [f"item_{i+1}" for i in range(len(norm_df))]

        for m in metrics:
            if m in norm_df.columns:
                norm_df[f"{mname}_{m}"] = pd.to_numeric(norm_df[m], errors="coerce")
            r_col = m.replace("_score", "_reasoning")
            if r_col in norm_df.columns:
                norm_df[f"{mname}_{r_col}"] = norm_df[r_col]

        dfs.append((mname, norm_df))

    base_df = dfs[0][1][["query_id", "query_text", "ground_truth"]].copy()
    for mname, mdf in dfs:
        cols_to_add = [c for c in mdf.columns if c.startswith(f"{mname}_")]
        base_df = base_df.merge(mdf[["query_id"] + cols_to_add], on="query_id", how="inner")

    candidate_models = [m for m in args.model_names if m != args.reference_model]

    all_metric_rankings = {}
    full_markdown_reports = []

    full_markdown_reports.append(f"# 📊 Laporan Benchmarking Model Evaluator RAGAS")
    full_markdown_reports.append(f"**Model Referensi Acuan (Anchor / Gold Standard):** `{args.reference_model}`\n")

    for metric in metrics:
        metric_label = metric.replace("_score", "").replace("_", " ").title()
        log_print(f"\n▶ EVALUASI KOMPARATIF TERHADAP METRIK: {metric_label}")
        log_print("-" * 65)

        benchmark_items = []
        for cand in candidate_models:
            res = evaluate_against_reference(
                df_merged=base_df,
                candidate_model=cand,
                reference_model=args.reference_model,
                metric_name=metric,
                threshold=args.threshold,
            )
            if res.get("status") == "success":
                benchmark_items.append(res)

        if not benchmark_items:
            log_print(f"  [Peringatan] Tidak ada data komparasi valid untuk {metric_label}")
            continue

        ranked_df, narrative = rank_models_and_generate_reasons(benchmark_items, args.reference_model)

        # Cetak tabel ringkasan ke log/terminal
        display_cols = [
            "candidate", "mean_candidate", "mean_bias", "mae", "pearson_r", "spearman_rho", "cohen_kappa", "tendency", "composite_score"
        ]
        log_print(ranked_df[display_cols].to_string(index=False))
        log_print("\n" + narrative + "\n")

        # Simpan CSV hasil metrik
        metric_csv = os.path.join(args.output_dir, f"benchmark_{metric}.csv")
        ranked_df.to_csv(metric_csv, index=False)

        # Tambahkan ke Markdown
        full_markdown_reports.append(f"## Metrik: {metric_label}")
        full_markdown_reports.append(ranked_df[display_cols].to_markdown(index=False))
        full_markdown_reports.append("\n" + narrative + "\n")

    # Tulis laporan markdown lengkap
    md_report_path = os.path.join(args.output_dir, "benchmark_recommendation_report.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_markdown_reports))

    log_print("==================================================================")
    log_print(f"✅ Seluruh hasil benchmarking berhasil disimpan:")
    log_print(f"   - Berkas CSV Ringkasan : {args.output_dir}/")
    log_print(f"   - Laporan Markdown     : {md_report_path}")
    log_print(f"   - Log Bukti Eksekusi   : {bench_log_path}")
    log_print("==================================================================")
    log_fp.close()


if __name__ == "__main__":
    main()
