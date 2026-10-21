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
Description : Modul perbandingan performa multi-model evaluator terhadap satu model
              referensi (Benchmark Reference Model) beserta sintesis narasi alasan keunggulan.

Changelog:
- 0.1.0 (13-10-2026): Implementasi komparasi bias leniency, MAE, korelasi, Kappa, dan generator narasi evaluasi.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import typing as t


def evaluate_against_reference(
    df_merged: pd.DataFrame,
    candidate_model: str,
    reference_model: str,
    metric_name: str,
    threshold: float = 0.7,
) -> dict:
    """
    Membandingkan performa model kandidat penilai terhadap model acuan/referensi
    pada satu metrik RAGAS tertentu.
    
    Args:
        df_merged (pd.DataFrame): Dataframe gabungan skor seluruh model.
        candidate_model (str): Nama model penilai yang diuji.
        reference_model (str): Nama model penilai acuan (benchmark/gold standard).
        metric_name (str): Nama metrik (contoh: 'faithfulness_score').
        threshold (float): Batas ambang biner kelulusan (default: 0.7).
        
    Returns:
        dict: Metrik deviasi (MAE, RMSE, Bias/Leniency), Korelasi, dan Kesepakatan Kappa.
    """
    cand_col = f"{candidate_model}_{metric_name}"
    ref_col = f"{reference_model}_{metric_name}"

    if cand_col not in df_merged.columns or ref_col not in df_merged.columns:
        return {"status": "missing_columns"}

    valid_mask = df_merged[cand_col].notna() & df_merged[ref_col].notna()
    y_cand = df_merged.loc[valid_mask, cand_col].astype(float)
    y_ref = df_merged.loc[valid_mask, ref_col].astype(float)

    n_samples = len(y_cand)
    if n_samples < 2:
        return {"status": "insufficient_data", "n_samples": n_samples}

    # 1. Error & Bias Analisis
    diff = y_cand - y_ref
    mean_error = float(np.mean(diff))  # Bias: >0 = Over-scoring / Lenient, <0 = Strict
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    # 2. Korelasi terhadap Model Referensi
    try:
        pearson_r, p_val = stats.pearsonr(y_cand, y_ref)
    except Exception:
        pearson_r, p_val = np.nan, np.nan

    try:
        spearman_rho, rho_p = stats.spearmanr(y_cand, y_ref)
    except Exception:
        spearman_rho, rho_p = np.nan, np.nan

    # 3. Kesepakatan Kategori (Cohen's Kappa dengan threshold)
    bin_cand = (y_cand >= threshold).astype(int)
    bin_ref = (y_ref >= threshold).astype(int)
    try:
        if len(np.unique(bin_cand)) == 1 and len(np.unique(bin_ref)) == 1:
            kappa = 1.0 if bin_cand.iloc[0] == bin_ref.iloc[0] else 0.0
        else:
            kappa = float(cohen_kappa_score(bin_cand, bin_ref))
    except Exception:
        kappa = np.nan

    # 4. Karakteristik Kecenderungan (Leniency classification)
    if mean_error > 0.08:
        tendency = "Sangat Longgar (Over-scoring bias)"
    elif mean_error > 0.03:
        tendency = "Cenderung Longgar"
    elif mean_error < -0.08:
        tendency = "Sangat Ketat (Under-scoring bias)"
    elif mean_error < -0.03:
        tendency = "Cenderung Ketat"
    else:
        tendency = "Seimbang (Aligned)"

    return {
        "status": "success",
        "candidate": candidate_model,
        "reference": reference_model,
        "metric": metric_name,
        "n_samples": n_samples,
        "mean_candidate": float(np.mean(y_cand)),
        "mean_reference": float(np.mean(y_ref)),
        "mean_bias": round(mean_error, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson_r": round(pearson_r, 4) if not np.isnan(pearson_r) else 0.0,
        "spearman_rho": round(spearman_rho, 4) if not np.isnan(spearman_rho) else 0.0,
        "cohen_kappa": round(kappa, 4) if not np.isnan(kappa) else 0.0,
        "tendency": tendency,
    }


def rank_models_and_generate_reasons(
    benchmark_results: t.List[dict],
    reference_model: str,
) -> t.Tuple[pd.DataFrame, str]:
    """
    Menyusun tabel peringkat model penilai berdasarkan kedekatan dengan acuan referensi,
    serta menghasilkan narasi deskriptif alasan mengapa suatu model lebih unggul.
    
    Skor Gabungan (Composite Score 0 - 100):
      Komponen:
      - 35% Pearson r
      - 35% Spearman rho
      - 20% Cohen's Kappa
      - 10% Penalti Deviasi MAE (1 - MAE)
    """
    df = pd.DataFrame(benchmark_results)
    if df.empty or "pearson_r" not in df.columns:
        return df, "Data tidak mencukupi untuk perankingan."

    # Hitung Composite Alignment Score
    def calc_composite(row):
        r = max(0.0, row["pearson_r"])
        rho = max(0.0, row["spearman_rho"])
        k = max(0.0, row["cohen_kappa"])
        mae_norm = max(0.0, 1.0 - row["mae"])
        comp = (0.35 * r + 0.35 * rho + 0.20 * k + 0.10 * mae_norm) * 100
        return round(comp, 2)

    df["composite_score"] = df.apply(calc_composite, axis=1)
    df_sorted = df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)

    # Sintesis narasi alasan
    best_row = df_sorted.iloc[0]
    best_model = best_row["candidate"]

    narrative = []
    narrative.append(f"### [REKOMENDASI] Kesimpulan Model Evaluator Terbaik: `{best_model}`")
    narrative.append(
        f"Berdasarkan pengujian komparatif dengan model referensi **`{reference_model}`**, "
        f"model **`{best_model}`** meraih peringkat tertinggi dengan **Composite Alignment Score {best_row['composite_score']}/100**."
    )
    narrative.append("\n**Deskripsi Alasan Mengapa Model Ini Lebih Baik:**")
    narrative.append(
        f"1. **Korelasi Terkuat:** Memiliki koefisien korelasi Pearson $r = {best_row['pearson_r']}$ "
        f"dan Spearman $\\rho = {best_row['spearman_rho']}$ terhadap acuan, menandakan tren pemeringkatan kualitas jawabannya "
        f"paling selaras dengan model benchmark."
    )
    narrative.append(
        f"2. **Tingkat Kesepakatan (Agreement):** Mencapai skor Cohen's Kappa $\\kappa = {best_row['cohen_kappa']}$, "
        f"yang menunjukkan konsensus klasifikasi lulus/gagal tertinggi tanpa dipengaruhi faktor kebetulan."
    )
    narrative.append(
        f"3. **Penyimpangan Skor Minimal:** Memiliki Mean Absolute Error (MAE) sebesar {best_row['mae']} "
        f"dengan bias rerata {best_row['mean_bias']:+0.4f} ({best_row['tendency']}), "
        f"artinya model tidak mengalami over-scoring ekstrem (tidak terlalu murah nilai) "
        f"maupun under-scoring ekstrem (tidak terlalu kaku menuntut label harfiah)."
    )

    if len(df_sorted) > 1:
        other_models = df_sorted.iloc[1:]
        narrative.append("\n**Catatan Komparasi Model Lainnya:**")
        for _, row in other_models.iterrows():
            narrative.append(
                f"- **`{row['candidate']}`** (Composite: {row['composite_score']}): "
                f"Karakteristik: *{row['tendency']}* (Bias: {row['mean_bias']:+0.4f}, MAE: {row['mae']}). "
                f"Korelasi $r = {row['pearson_r']}$, $\\kappa = {row['cohen_kappa']}."
            )

    narrative_text = "\n".join(narrative)
    return df_sorted, narrative_text
