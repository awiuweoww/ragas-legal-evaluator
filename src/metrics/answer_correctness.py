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

import numpy as np
from ragas.metrics._answer_correctness import (
    AnswerCorrectness,
    QuestionAnswerGroundTruth,
    AnswerSimilarity,
)
from ragas.dataset_schema import BaseSample, SingleTurnSample


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

        question = s.get("user_input", "") or s.get("question", "") or ""
        response = s.get("response", "") or s.get("answer", "") or ""
        reference = s.get("reference", "") or s.get("ground_truth", "") or ""

        # Inisialisasi semantic similarity jika belum
        if self.answer_similarity is None and self.weights[1] != 0:
            if self.embeddings is not None:
                self.answer_similarity = AnswerSimilarity(embeddings=self.embeddings)

        # 1. Ekstraksi simplified statements dari response dan reference
        statements: t.Dict[str, t.List[str]] = {}
        for item_key, text in [("response", response), ("reference", reference)]:
            try:
                stmts_out = await self._create_simplified_statements(question, text, callbacks=None)
                statements[item_key] = (
                    stmts_out.statements if hasattr(stmts_out, "statements") else (stmts_out or [])
                )
            except Exception:
                statements[item_key] = []

        # 2. Klasifikasi TP, FP, FN
        answers_obj = None
        f1_score = 0.0
        if statements.get("response") or statements.get("reference"):
            try:
                answers_obj = await self.correctness_prompt.generate(
                    llm=self.llm,
                    data=QuestionAnswerGroundTruth(
                        question=question,
                        answer=statements.get("response", []),
                        ground_truth=statements.get("reference", []),
                    ),
                    callbacks=None,
                )
                if answers_obj is not None:
                    f1_score = float(self._compute_statement_presence(answers_obj))
            except Exception:
                f1_score = 0.0
        else:
            f1_score = 1.0

        # 3. Kesamaan semantik (Embeddings)
        similarity_score = 0.0
        if self.weights[1] != 0 and self.answer_similarity is not None and response and reference:
            try:
                row_data = {"reference": reference, "response": response}
                sim_res = await self.answer_similarity._ascore(row_data, callbacks=None)
                if isinstance(sim_res, (np.ndarray, list)):
                    sim_res = float(sim_res[0])
                similarity_score = float(sim_res) if sim_res is not None and not np.isnan(sim_res) else 0.0
            except Exception as e:
                similarity_score = 0.0

        # 4. Harmonisasi skor akhir
        final_score = float(np.average([f1_score, similarity_score], weights=self.weights))

        self.last_results = {
            "score": final_score,
            "answers": answers_obj,
            "similarity_score": similarity_score,
            "f1_score": f1_score,
        }
        return final_score

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
