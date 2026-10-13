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
Description : CLI script eksekusi analisis komparasi statistik antar-model evaluator RAGAS:
              Pearson r, Spearman rho, Cohen's Kappa, Fleiss' Kappa, ICC, dan Alignment Sampling.

Changelog:
- 0.1.0 (13-10-2026): Implementasi komputasi matriks otomatis dan ekspor ringkasan CSV.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from src.analysis.correlation import calculate_pairwise_correlation, compute_correlation_matrices
from src.analysis.agreement import calculate_cohen_kappa, calculate_fleiss_kappa, calculate_icc
from src.analysis.alignment import find_high_variance_cases


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisasi ragam nama kolom dari format log RAGAS lama maupun skema eksperimen baru.
    """
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
    parser = argparse.ArgumentParser(description="Analisis Statistik Korelasi & Agreement Antar-Model Evaluator")
    parser.add_argument("--input_files", nargs="+", required=True, help="Daftar file CSV hasil evaluasi per model")
    parser.add_argument("--model_names", nargs="+", required=True, help="Nama identitas masing-masing model")
    parser.add_argument("--output_dir", default="data/results/statistical_summary", help="Folder output")
    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold biner untuk Cohen & Fleiss Kappa")
    args = parser.parse_args()

    if len(args.input_files) != len(args.model_names):
        raise ValueError("Jumlah input_files harus sama persis dengan jumlah model_names!")

    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    analysis_log_path = os.path.join(logs_dir, f"analysis_{'_'.join(args.model_names)}.log")
    log_fp = open(analysis_log_path, "w", encoding="utf-8")

    def log_print(msg=""):
        print(msg)
        log_fp.write(str(msg) + "\n")
        log_fp.flush()

    metrics = ["faithfulness_score", "context_recall_score", "answer_correctness_score"]

    log_print(f"\n=======================================================")
    log_print(f"📊 EKSEKUSI ANALISIS STATISTIK EVALUATOR RAGAS (Tugas 2b)")
    log_print(f"Model Penilai   : {args.model_names}")
    log_print(f"Threshold Kappa : >= {args.threshold}")
    log_print(f"Berkas Log Bukti: {analysis_log_path}")
    log_print(f"=======================================================\n")

    # Muat dan gabungkan DataFrame berdasarkan query_id atau indeks
    dfs = []
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

    for metric in metrics:
        metric_label = metric.replace("_score", "").replace("_", " ").title()
        log_print(f"\n▶ METRIK: {metric_label}")
        log_print("-" * 50)

        score_cols = [f"{m}_{metric}" for m in args.model_names if f"{m}_{metric}" in base_df.columns]
        if len(score_cols) < 2:
            log_print(f"  [Lewati] Tidak cukup kolom skor untuk {metric}")
            continue

        # 1. Parameter 2: Korelasi Antar-Model (Pearson & Spearman)
        pearson_mat, spearman_mat = compute_correlation_matrices(base_df, args.model_names, metric)
        log_print("\n[1] Pearson Correlation Matrix (r):")
        log_print(pearson_mat.round(4).to_string())
        pearson_mat.to_csv(os.path.join(args.output_dir, f"pearson_{metric}.csv"))

        log_print("\n[2] Spearman Rank Correlation Matrix (rho):")
        log_print(spearman_mat.round(4).to_string())
        spearman_mat.to_csv(os.path.join(args.output_dir, f"spearman_{metric}.csv"))

        # 2. Parameter 3: Cohen's Kappa Pairwise
        log_print("\n[3] Cohen's Kappa Pairwise (Threshold >= {}):".format(args.threshold))
        kappa_matrix = pd.DataFrame(index=args.model_names, columns=args.model_names, dtype=float)
        for i, m1 in enumerate(args.model_names):
            for j, m2 in enumerate(args.model_names):
                if i == j:
                    kappa_matrix.loc[m1, m2] = 1.0
                else:
                    k_res = calculate_cohen_kappa(base_df[f"{m1}_{metric}"], base_df[f"{m2}_{metric}"], args.threshold)
                    kappa_matrix.loc[m1, m2] = k_res["cohen_kappa"]
        log_print(kappa_matrix.round(4).to_string())
        kappa_matrix.to_csv(os.path.join(args.output_dir, f"cohen_kappa_{metric}.csv"))

        # 3. Parameter 3: Fleiss' Kappa (> 2 Model)
        if len(args.model_names) > 2:
            bin_ratings = (base_df[score_cols] >= args.threshold).astype(int).values
            cat_counts = np.zeros((len(bin_ratings), 2), dtype=int)
            for row_idx, row in enumerate(bin_ratings):
                cat_counts[row_idx, 0] = np.sum(row == 0)
                cat_counts[row_idx, 1] = np.sum(row == 1)
            
            fleiss_k = calculate_fleiss_kappa(cat_counts)
            log_print(f"\n[4] Fleiss' Kappa (Seluruh {len(args.model_names)} Model): {fleiss_k:.4f}")

        # 4. Parameter 3: Intraclass Correlation Coefficient (ICC)
        icc_res = calculate_icc(base_df, score_cols)
        log_print(f"[5] Intraclass Correlation Coefficient (ICC): {icc_res['icc']:.4f} (Metode: {icc_res['method']})")

        # 5. Parameter 1: Alignment Sampling (High Variance Cases)
        high_var = find_high_variance_cases(base_df, score_cols, top_n=5)
        out_cases = high_var[["query_id", "query_text", "score_std"] + score_cols]
        out_cases.to_csv(os.path.join(args.output_dir, f"high_variance_{metric}.csv"), index=False)
        log_print(f"\n[6] 5 Kasus Variansi/Disagreement Tertinggi disimpan di: high_variance_{metric}.csv")

    log_print(f"\nSeluruh ringkasan tabel berhasil disimpan di direktori: {args.output_dir}\n")
    log_fp.close()


if __name__ == "__main__":
    main()
