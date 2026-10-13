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
Description : Modul analisis statistik korelasi antar-model evaluator RAGAS
              (Pearson Correlation Coefficient dan Spearman's Rank Correlation).

Changelog:
- 0.1.0 (13-10-2026): Implementasi fungsi pairwise correlation dan matrix correlation.
"""

import numpy as np
import pandas as pd
from scipy import stats
import typing as t


def calculate_pairwise_correlation(
    scores_a: pd.Series, scores_b: pd.Series
) -> dict:
    """
    Menghitung Pearson Correlation (r) dan Spearman Rank Correlation (rho)
    antara dua set skor numerik evaluator.
    
    - Pearson r: Mengukur korelasi linear antar-skor kontinu.
    - Spearman rho: Mengukur konsistensi peringkat/urutan (rank consistency) non-parametrik.
    
    Args:
        scores_a (pd.Series): Seri skor dari model evaluator A.
        scores_b (pd.Series): Seri skor dari model evaluator B.
        
    Returns:
        dict: Berisi n_samples, pearson_r, pearson_p, spearman_rho, spearman_p.
    """
    # Filter nilai missing (NaN) dan pastikan tipe data numerik
    valid_mask = scores_a.notna() & scores_b.notna()
    a_clean = scores_a[valid_mask].astype(float)
    b_clean = scores_b[valid_mask].astype(float)

    if len(a_clean) < 2:
        return {
            "n_samples": len(a_clean),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
        }

    # Pearson Correlation Coefficient
    try:
        pearson_r, pearson_p = stats.pearsonr(a_clean, b_clean)
    except Exception:
        pearson_r, pearson_p = np.nan, np.nan

    # Spearman Rank Correlation Coefficient
    try:
        spearman_rho, spearman_p = stats.spearmanr(a_clean, b_clean)
    except Exception:
        spearman_rho, spearman_p = np.nan, np.nan

    return {
        "n_samples": len(a_clean),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def compute_correlation_matrices(
    df_evals: pd.DataFrame, models: t.List[str], score_col: str
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menghasilkan matriks simetris Pearson dan Spearman untuk seluruh pasangan model evaluator.
    
    Args:
        df_evals (pd.DataFrame): DataFrame gabungan berisi skor seluruh model.
        models (List[str]): Daftar nama model penilai.
        score_col (str): Nama metrik skor yang diuji (misal: 'faithfulness_score').
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (pearson_matrix, spearman_matrix)
    """
    pearson_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    spearman_matrix = pd.DataFrame(index=models, columns=models, dtype=float)

    for m1 in models:
        for m2 in models:
            if m1 == m2:
                pearson_matrix.loc[m1, m2] = 1.0
                spearman_matrix.loc[m1, m2] = 1.0
            else:
                col1 = f"{m1}_{score_col}" if f"{m1}_{score_col}" in df_evals else m1
                col2 = f"{m2}_{score_col}" if f"{m2}_{score_col}" in df_evals else m2
                res = calculate_pairwise_correlation(df_evals[col1], df_evals[col2])
                pearson_matrix.loc[m1, m2] = res["pearson_r"]
                spearman_matrix.loc[m1, m2] = res["spearman_rho"]

    return pearson_matrix, spearman_matrix
