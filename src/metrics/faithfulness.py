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
Description : Modul metrik kustom Faithfulness RAGAS dengan ekstraksi
              output reasoning eksplisit dan verdict per statement.

Changelog:
- 0.1.0 (13-10-2026): Inisialisasi awal FaithfulnessExtractor dengan retries dan structured JSON extraction.
"""

import asyncio
import json
import typing as t
from dataclasses import dataclass

from ragas.metrics._faithfulness import Faithfulness
from ragas.dataset_schema import BaseSample


@dataclass
class SingleTurnSample(BaseSample):
    """
    Representasi sample tunggal untuk proses evaluasi metrik Faithfulness.
    Menampung input query pengguna, respons LLM generator, dan retrieved context.
    """
    user_input: t.Optional[str] = None
    question: t.Optional[str] = None
    response: t.Optional[str] = None
    answer: t.Optional[str] = None
    retrieved_contexts: t.Optional[t.List[str]] = None
    contexts: t.Optional[t.List[str]] = None


class FaithfulnessExtractor(Faithfulness):
    """
    Subclass turunan dari Faithfulness RAGAS yang menyimpan teks penalaran (reasoning),
    pernyataan terurai (statements), dan vonis NLI (verdicts) secara terstruktur.
    
    Metrik ini mengukur sejauh mana klaim/fakta dalam jawaban sistem didukung secara
    eksplisit oleh konteks dokumen yang di-retrieve, guna mendeteksi halusinasi.
    """

    def __init__(self, llm_callable=None):
        super().__init__(llm=llm_callable)
        self.verdicts = None
        self.statements = None

    async def _ascore(self, sample: t.Union[dict, SingleTurnSample]) -> float:
        """
        Menghitung skor Faithfulness secara asinkron dengan mekanisme toleransi kegagalan (retry).
        """
        if isinstance(sample, dict):
            s = sample
        else:
            s = {
                "user_input": getattr(sample, "user_input", None),
                "response": getattr(sample, "response", None),
                "retrieved_contexts": getattr(sample, "retrieved_contexts", None),
            }

        # Langkah 1: Ekstraksi klaim atau proposisi mandiri dari respons model
        try:
            raw_statements = await self._create_statements(s, callbacks=None)
        except Exception:
            try:
                raw_statements = await self._create_statements(s, callbacks=None)
            except Exception as e2:
                raise RuntimeError(f"Gagal mengekstrak statements: {e2}") from e2

        self.statements = raw_statements

        # Ekstrak list string statements murni untuk NLI
        if hasattr(raw_statements, "statements"):
            stmts_list = raw_statements.statements
        elif isinstance(raw_statements, dict):
            stmts_list = raw_statements.get("statements", [])
        elif isinstance(raw_statements, list):
            stmts_list = raw_statements
        else:
            stmts_list = []

        # Jika respons model tidak membuat klaim faktual, skor dianggap 1.0 (tidak ada halusinasi)
        if not stmts_list:
            self.verdicts = None
            return 1.0

        # Langkah 2: NLI Verifikasi terhadap konteks rujukan
        try:
            raw_verdicts = await self._create_verdicts(s, stmts_list, callbacks=None)
        except Exception:
            try:
                raw_verdicts = await self._create_verdicts(s, stmts_list, callbacks=None)
            except Exception as e2:
                raise RuntimeError(f"Gagal mengekstrak verdicts: {e2}") from e2

        self.verdicts = raw_verdicts
        return float(self._compute_score(raw_verdicts))

    def get_last_results(self) -> dict:
        """Mengembalikan rincian data mentah statements dan verdicts evaluasi terakhir."""
        return {
            "statements": self.statements,
            "verdicts": self.verdicts,
        }

    @staticmethod
    def extract_reasoning(results: dict) -> t.Tuple[str, list]:
        """
        Mengekstrak teks penalaran logis manusiawi dan daftar per-item klaim:
        Format: [OK|FAIL] <pernyataan>: <alasan model juri>
        """
        verdicts_obj = results.get("verdicts")
        items = []
        if verdicts_obj is not None:
            raw_list = []
            if hasattr(verdicts_obj, "statements"):
                raw_list = verdicts_obj.statements
            elif isinstance(verdicts_obj, dict):
                raw_list = verdicts_obj.get("statements", [])
            elif isinstance(verdicts_obj, list):
                raw_list = verdicts_obj

            for ans in raw_list:
                stmt = getattr(ans, "statement", None) if not isinstance(ans, dict) else ans.get("statement")
                verdict = getattr(ans, "verdict", None) if not isinstance(ans, dict) else ans.get("verdict")
                reason = getattr(ans, "reason", None) if not isinstance(ans, dict) else ans.get("reason")
                if stmt is not None:
                    items.append({
                        "statement": stmt,
                        "verdict": int(verdict) if verdict is not None else 0,
                        "reason": reason or "",
                    })

        explanation = " | ".join(
            f"[{'OK' if it['verdict'] else 'FAIL'}] {it['statement']}: {it['reason']}"
            for it in items
            if it.get("statement")
        )
        return explanation, items
