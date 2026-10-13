"""
Copyright (c) 2026 Badan Riset dan Inovasi Nasional (BRIN)
THIS SOFTWARE SOURCE CODE AND ANY EXECUTABLE DERIVED THEREOF ARE PROPRIETARY
TO BRIN, AS APPLICABLE, AND SHALL NOT BE USED IN ANY WAY
OTHER THAN BEFOREHAND AGREED ON BY BRIN, NOR BE REPRODUCED
OR DISCLOSED TO THIRD PARTIES WITHOUT PRIOR WRITTEN AUTHORIZATION BY
BRIN, AS APPLICABLE.

Author      : Chandra
Version     : 0.1.1
Created     : 13-10-2026
Description : Package inisialisasi modul analisis statistik korelasi,
              agreement reliability, alignment reasoning, dan benchmarking model.

Changelog:
- 0.1.0 (13-10-2026): Ekspor fungsi korelasi Pearson/Spearman, Kappa, ICC, dan outlier analysis.
- 0.1.1 (13-10-2026): Penambahan fungsi komparasi dan perankingan model terhadap model referensi.
"""

from .correlation import calculate_pairwise_correlation, compute_correlation_matrices
from .agreement import calculate_cohen_kappa, calculate_fleiss_kappa, calculate_icc
from .alignment import find_high_variance_cases, compare_reasoning_for_sample
from .benchmarking import evaluate_against_reference, rank_models_and_generate_reasons

__all__ = [
    "calculate_pairwise_correlation",
    "compute_correlation_matrices",
    "calculate_cohen_kappa",
    "calculate_fleiss_kappa",
    "calculate_icc",
    "find_high_variance_cases",
    "compare_reasoning_for_sample",
    "evaluate_against_reference",
    "rank_models_and_generate_reasons",
]
