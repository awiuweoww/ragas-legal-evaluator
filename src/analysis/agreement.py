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
Description : Modul analisis statistik tingkat kesepakatan (Agreement Score)
              antar-model penilai: Cohen's Kappa, Fleiss' Kappa, dan ICC.

Changelog:
- 0.1.0 (13-10-2026): Implementasi Cohen's Kappa biner, Fleiss' Kappa multi-rater, dan ICC(2,1).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import typing as t


def calculate_cohen_kappa(
    scores_a: pd.Series, scores_b: pd.Series, threshold: float = 0.7
) -> dict:
    """
    Menghitung Cohen's Kappa untuk mengukur tingkat kesepakatan kategori biner
    antara dua model penilai (pairwise) dengan mengompensasi faktor kebetulan (chance agreement).
    
    Kategori ditentukan berdasarkan threshold:
    - 1: Lolos / Berkualitas Tinggi (skor >= threshold)
    - 0: Gagal / Kurang Berkualitas (skor < threshold)
    
    Args:
        scores_a (pd.Series): Skor model A.
        scores_b (pd.Series): Skor model B.
        threshold (float): Batas ambang kelulusan (default: 0.7).
        
    Returns:
        dict: Hasil perhitungan kappa, jumlah sampel valid, dan threshold.
    """
    valid_mask = scores_a.notna() & scores_b.notna()
    a_clean = scores_a[valid_mask].astype(float)
    b_clean = scores_b[valid_mask].astype(float)

    if len(a_clean) == 0:
        return {"n_samples": 0, "cohen_kappa": np.nan, "threshold": threshold}

    bin_a = (a_clean >= threshold).astype(int)
    bin_b = (b_clean >= threshold).astype(int)

    # Tangani kasus khusus jika seluruh data berada dalam satu kategori yang sama
    if len(np.unique(bin_a)) == 1 and len(np.unique(bin_b)) == 1:
        kappa = 1.0 if bin_a.iloc[0] == bin_b.iloc[0] else 0.0
    else:
        kappa = cohen_kappa_score(bin_a, bin_b)

    return {
        "n_samples": len(a_clean),
        "cohen_kappa": float(kappa),
        "threshold": threshold,
    }


def calculate_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Menghitung Fleiss' Kappa untuk mengukur kesepakatan serentak antar > 2 model penilai.
    
    Args:
        ratings_matrix (np.ndarray): Matriks ukuran (N, k):
            N = jumlah pertanyaan/kasus sampel
            k = jumlah kategori (misal: 2 kategori -> [Gagal, Lolos])
            Nilai elemen [i, j] adalah jumlah penilai yang memasukkan subjek i ke kategori j.
            
    Returns:
        float: Nilai Fleiss' Kappa (-1.0 hingga +1.0).
    """
    N, k = ratings_matrix.shape
    n = np.sum(ratings_matrix[0, :])  # jumlah penilai tetap per sampel

    if n <= 1:
        return np.nan

    # Proporsi kategori j di seluruh penilai
    p_j = np.sum(ratings_matrix, axis=0) / (N * n)
    p_e = np.sum(p_j**2)

    # Derajat kesepakatan internal pada setiap sampel i
    P_i = (np.sum(ratings_matrix**2, axis=1) - n) / (n * (n - 1))
    P_mean = np.mean(P_i)

    if 1.0 - p_e == 0:
        return 1.0 if P_mean == 1.0 else 0.0

    kappa = (P_mean - p_e) / (1.0 - p_e)
    return float(kappa)


def calculate_icc(df_wide: pd.DataFrame, score_cols: t.List[str]) -> dict:
    """
    Menghitung Intraclass Correlation Coefficient (ICC(2,1)) untuk reliabilitas data kontinu.
    Menggunakan library pingouin jika tersedia, dengan fallback perhitungan Two-Way Random ANOVA.
    
    Args:
        df_wide (pd.DataFrame): DataFrame berisi kolom skor masing-masing model penilai.
        score_cols (List[str]): Nama-nama kolom skor model penilai.
        
    Returns:
        dict: Berisi nilai ICC, tipe ICC, dan metode komputasi yang dipakai.
    """
    df_clean = df_wide[score_cols].dropna().astype(float)
    if len(df_clean) < 2 or len(score_cols) < 2:
        return {"icc": np.nan, "icc_type": "ICC(2,1)", "method": "insufficient_data"}

    try:
        import pingouin as pg

        # Ubah wide DataFrame menjadi format long (target, rater, score)
        df_long = df_clean.reset_index().melt(
            id_vars=["index"],
            value_vars=score_cols,
            var_name="rater",
            value_name="score",
        )
        icc_res = pg.intraclass_corr(
            data=df_long, targets="index", raters="rater", ratings="score"
        )
        icc2 = icc_res[icc_res["Type"] == "ICC2"].iloc[0]["ICC"]
        return {"icc": float(icc2), "icc_type": "ICC(2,1)", "method": "pingouin"}
    except Exception:
        # Fallback algoritma Two-Way Random Effects ANOVA (ICC(2,1))
        Y = df_clean.values
        n, k = Y.shape
        grand_mean = np.mean(Y)
        target_means = np.mean(Y, axis=1)
        rater_means = np.mean(Y, axis=0)

        sst = np.sum((Y - grand_mean) ** 2)
        ssr = k * np.sum((target_means - grand_mean) ** 2)
        ssc = n * np.sum((rater_means - grand_mean) ** 2)
        sse = sst - ssr - ssc

        msr = ssr / (n - 1)
        msc = ssc / (k - 1)
        mse = sse / ((n - 1) * (k - 1))

        denominator = msr + (k - 1) * mse + (k / n) * (msc - mse)
        icc21 = (msr - mse) / denominator if denominator != 0 else np.nan

        return {"icc": float(icc21), "icc_type": "ICC(2,1)", "method": "native_anova"}
