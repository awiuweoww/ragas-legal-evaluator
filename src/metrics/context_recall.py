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
Description : Modul metrik kustom Context Recall RAGAS dengan ekstraksi
              alasan atribusi (reasoning) pada setiap kalimat Ground Truth.

Changelog:
- 0.1.0 (13-10-2026): Inisialisasi awal ContextRecallExtractor dengan tracking atribusi per klausul/pasal.
"""

import asyncio
import json
import typing as t
from dataclasses import dataclass

from ragas.metrics._context_recall import (
    LLMContextRecall,
    QCA,
    ContextRecallClassification,
    ContextRecallClassifications,
    ensembler,
)
from ragas.dataset_schema import BaseSample


@dataclass
class SingleTurnSample(BaseSample):
    """
    Representasi sample evaluasi metrik Context Recall.
    Context Recall hanya membandingkan ground truth (reference) terhadap retrieved context.
    """
    user_input: t.Optional[str] = None
    question: t.Optional[str] = None
    reference: t.Optional[str] = None
    ground_truth: t.Optional[str] = None
    retrieved_contexts: t.Optional[t.List[str]] = None
    contexts: t.Optional[t.List[str]] = None


class ContextRecallExtractor(LLMContextRecall):
    """
    Subclass turunan dari LLMContextRecall RAGAS yang menyimpan klasifikasi atribusi
    beserta justifikasi logis pada setiap butir pernyataan kunci jawaban (ground truth).
    
    Metrik ini mengukur kelengkapan dokumen hukum yang berhasil di-retrieve sistem:
    Apakah semua unsur hukum yang dibutuhkan untuk menjawab pertanyaan berhasil ditemukan?
    """

    def __init__(self, llm_callable=None):
        super().__init__(llm=llm_callable)
        self.classifications = None

    async def _ascore(self, sample: t.Union[dict, SingleTurnSample]) -> float:
        """
        Menjalankan klasifikasi atribusi secara asinkron dengan penanganan error.
        """
        if isinstance(sample, dict):
            s = sample
        else:
            s = {
                "user_input": getattr(sample, "user_input", None),
                "reference": getattr(sample, "reference", None),
                "retrieved_contexts": getattr(sample, "retrieved_contexts", None),
            }

        question = s.get("user_input", "") or s.get("question", "") or ""
        contexts = s.get("retrieved_contexts", []) or s.get("contexts", []) or []
        context_str = "\n".join(contexts) if isinstance(contexts, list) else str(contexts)
        reference = s.get("reference", "") or s.get("ground_truth", "") or ""

        qca_data = QCA(
            question=question,
            context=context_str,
            answer=reference,
        )

        try:
            classifications_list = await self.context_recall_prompt.generate_multiple(
                data=qca_data,
                llm=self.llm,
                callbacks=None,
            )
        except Exception:
            try:
                classifications_list = await self.context_recall_prompt.generate_multiple(
                    data=qca_data,
                    llm=self.llm,
                    callbacks=None,
                )
            except Exception as e2:
                raise RuntimeError(f"Gagal mengekstrak context recall classifications: {e2}") from e2

        classification_dicts = []
        for classification in classifications_list:
            if hasattr(classification, "classifications"):
                classification_dicts.append(
                    [
                        clasif.model_dump() if hasattr(clasif, "model_dump") else clasif.__dict__
                        for clasif in classification.classifications
                    ]
                )

        if classification_dicts:
            ensembled_clasif = ensembler.from_discrete(classification_dicts, "attributed")
            items = [ContextRecallClassification(**clasif) for clasif in ensembled_clasif]
            score = self._compute_score(items)
            self.classifications = [
                c.model_dump() if hasattr(c, "model_dump") else c.__dict__ for c in items
            ]
        else:
            score = 0.0
            self.classifications = []

        return float(score)

    def get_last_results(self) -> dict:
        """Mengembalikan klasifikasi atribusi dari evaluasi terakhir."""
        return {"classifications": self.classifications}

    @staticmethod
    def extract_reasoning(results: dict) -> t.Tuple[str, list]:
        """
        Mengekstrak penjelasan logis:
        Format: [OK|FAIL] <klausul rujukan>: <alasan model juri apakah terdapat dalam konteks>
        """
        classifications = results.get("classifications") or []
        flat = []
        if isinstance(classifications, list):
            for item in classifications:
                if isinstance(item, dict):
                    flat.append(item)
                elif hasattr(item, "__dict__"):
                    flat.append(item.__dict__)

        explanation = " | ".join(
            f"[{'OK' if it.get('attributed') else 'FAIL'}] {it.get('statement')}: {it.get('reason')}"
            for it in flat
            if it.get("statement")
        )
        return explanation, flat
