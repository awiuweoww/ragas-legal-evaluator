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
Description : Modul metrik kustom Answer Correctness RAGAS dengan pemisahan
              komponen F1 faktual, similarity semantik, serta justifikasi TP/FP/FN.

Changelog:
- 0.1.0 (13-10-2026): Inisialisasi awal AnswerCorrectnessExtractor dengan tracking reasoning terperinci.
"""

import asyncio
import json
import typing as t
from dataclasses import dataclass

from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.dataset_schema import BaseSample


@dataclass
class SingleTurnSample(BaseSample):
    """
    Representasi sample evaluasi metrik Answer Correctness.
    Menghubungkan respons model generator terhadap kunci jawaban pakar (ground truth).
    """
    user_input: t.Optional[str] = None
    question: t.Optional[str] = None
    response: t.Optional[str] = None
    answer: t.Optional[str] = None
    reference: t.Optional[str] = None
    ground_truth: t.Optional[str] = None


class AnswerCorrectnessExtractor(AnswerCorrectness):
    """
    Subclass turunan dari AnswerCorrectness RAGAS.
    Mengekstrak perincian evaluasi faktual (True Positives, False Positives, False Negatives),
    skor kesamaan semantik (semantic similarity via embeddings), dan skor harmonisasi akhir.
    """

    def __init__(self, llm_callable=None, evaluator_embeddings=None):
        kwargs = {}
        if llm_callable is not None:
            kwargs["llm"] = llm_callable
        if evaluator_embeddings is not None:
            kwargs["embeddings"] = evaluator_embeddings
        super().__init__(**kwargs)
        self.last_results = {}

    async def _ascore(self, sample: t.Union[dict, SingleTurnSample]) -> float:
        """
        Menghitung akurasi jawaban secara asinkron dan menyimpan metadata breakdown.
        """
        if isinstance(sample, dict):
            s = sample
        else:
            s = {
                "user_input": getattr(sample, "user_input", None),
                "response": getattr(sample, "response", None),
                "reference": getattr(sample, "reference", None),
            }

        score = await super()._ascore(s)
        self.last_results = {
            "score": score,
            "answers": getattr(self, "answers", None),
            "similarity_score": getattr(self, "similarity_score", None),
            "f1_score": getattr(self, "f1_score", None),
        }
        return score

    def get_last_results(self) -> dict:
        """Mengembalikan rincian pembobotan faktual dan kesamaan semantik."""
        return self.last_results

    @staticmethod
    def extract_reasoning(results: dict) -> t.Tuple[str, dict]:
        """
        Mengekstrak justifikasi per klasifikasi:
        - [TP] Fakta benar yang sesuai ground truth
        - [FP] Fakta halusinasi/salah yang tidak ada di ground truth
        - [FN] Fakta penting di ground truth yang diabaikan model
        """
        answers = results.get("answers")
        details = {}
        if answers is not None:
            if hasattr(answers, "model_dump"):
                details = answers.model_dump()
            elif hasattr(answers, "dict"):
                details = answers.dict()
            elif isinstance(answers, dict):
                details = answers
            elif hasattr(answers, "__dict__"):
                details = answers.__dict__

        parts = []
        for label in ("TP", "FP", "FN"):
            for item in (details.get(label) or []):
                stmt = item.get("statement") if isinstance(item, dict) else getattr(item, "statement", None)
                reason = item.get("reason") if isinstance(item, dict) else getattr(item, "reason", None)
                parts.append(f"[{label}] {stmt}: {reason}")

        explanation = " | ".join(parts)
        return explanation, details
