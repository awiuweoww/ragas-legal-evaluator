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
Description : Modul analisis kualitatif penalaran (Alignment Reasoning) dan
              purposive sampling untuk mendeteksi deviasi/ketidaksepakatan tinggi.

Changelog:
- 0.1.0 (13-10-2026): Implementasi seleksi kasus variansi tinggi dan perbandingan teks reasoning.
"""

import numpy as np
import pandas as pd
import typing as t


def find_high_variance_cases(
    df_merged: pd.DataFrame,
    score_cols: t.List[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Menemukan sampel kasus hukum dengan variansi skor tertinggi antar-model penilai.
    Berguna untuk purposive sampling audit kualitatif guna memeriksa:
    1. Apakah ada model yang mengalami halusinasi kritik (hallucinated critique)?
    2. Apakah perbedaan skor timbul dari interpretasi hukum yang berbeda atau pemahaman pasal?
    
    Args:
        df_merged (pd.DataFrame): DataFrame berisi skor seluruh model.
        score_cols (List[str]): Daftar nama kolom skor model.
        top_n (int): Jumlah kasus teratas yang diambil.
        
    Returns:
        pd.DataFrame: Sub-dataset kasus dengan standar deviasi skor tertinggi.
    """
    df = df_merged.copy()
    numeric_scores = df[score_cols].apply(pd.to_numeric, errors="coerce")
    df["score_std"] = numeric_scores.std(axis=1)
    df["score_range"] = numeric_scores.max(axis=1) - numeric_scores.min(axis=1)

    sorted_df = df.sort_values(by="score_std", ascending=False)
    return sorted_df.head(top_n)


def compare_reasoning_for_sample(
    df_merged: pd.DataFrame,
    query_id: str,
    models: t.List[str],
    metric_name: str,
) -> dict:
    """
    Mengambil dan menyandingkan teks penalaran (reasoning) antar-model pada satu sampel kasus spesifik.
    
    Args:
        df_merged (pd.DataFrame): DataFrame gabungan evaluasi.
        query_id (str): Identitas query kasus.
        models (List[str]): Daftar nama model penilai.
        metric_name (str): Nama metrik yang diperiksa (contoh: 'faithfulness', 'context_recall').
        
    Returns:
        dict: Struktur ringkasan perbandingan teks alasan tiap model.
    """
    row = df_merged[df_merged["query_id"] == query_id]
    if row.empty:
        return {}

    row = row.iloc[0]
    comparison = {
        "query_id": query_id,
        "query_text": row.get("query_text", ""),
        "ground_truth": row.get("ground_truth", ""),
        "metric": metric_name,
        "evaluators": {},
    }

    for m in models:
        score_col = f"{m}_{metric_name}_score"
        reasoning_col = f"{m}_{metric_name}_reasoning"
        comparison["evaluators"][m] = {
            "score": row.get(score_col, np.nan),
            "reasoning": row.get(reasoning_col, ""),
        }

    return comparison
