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
Description : Pipeline eksekusi batch evaluasi RAGAS otomatis dengan penanganan
              interupsi (resume), streaming I/O CSV, logging audit per-sampel,
              dan pembuatan metadata JSON sebagai bukti eksperimen.

Changelog:
- 0.1.0 (13-10-2026): Implementasi EvaluationRunner dengan schema output komprehensif.
- 0.1.1 (13-10-2026): Penambahan audit logging otomatis per baris dan ekspor run metadata JSON.
"""

import asyncio
import csv
import json
import logging
import os
import sys
import time
import typing as t
from datetime import datetime
from tqdm import tqdm

from src.metrics.faithfulness import FaithfulnessExtractor
from src.metrics.context_recall import ContextRecallExtractor
from src.metrics.answer_correctness import AnswerCorrectnessExtractor
from src.evaluators.llm_factory import build_evaluator_llm, build_evaluator_embeddings
from src.utils.logger import format_sample_evaluation, AnsiColor

OUTPUT_COLUMNS = [
    "query_id",
    "query_text",
    "background_scenario",
    "ground_truth",
    "response_answer",
    "retrieved_contexts",
    "model_evaluator",
    # Metrik Faithfulness
    "faithfulness_score",
    "faithfulness_reasoning",
    "faithfulness_details_json",
    # Metrik Context Recall
    "context_recall_score",
    "context_recall_reasoning",
    "context_recall_details_json",
    # Metrik Answer Correctness
    "answer_correctness_score",
    "answer_correctness_f1",
    "answer_correctness_similarity",
    "answer_correctness_reasoning",
    "answer_correctness_details_json",
]


def truncate_context(ctx: str, max_chars: int = 270000) -> str:
    """
    Memangkas konteks yang sangat panjang agar tidak melampaui batas konteks LLM,
    dengan memprioritaskan bagian referensi pasal ('-----Sources-----').
    """
    if not ctx or len(ctx) <= max_chars:
        return ctx
    marker = "-----Sources-----"
    idx = ctx.find(marker)
    if idx == -1:
        return ctx[:max_chars]
    sources = ctx[idx:]
    if len(sources) >= max_chars:
        return sources[:max_chars]
    head_budget = max_chars - len(sources)
    return ctx[:head_budget] + "\n" + sources


async def evaluate_single_sample(
    sample_data: dict,
    fe: t.Optional[FaithfulnessExtractor],
    ce: t.Optional[ContextRecallExtractor],
    ae: t.Optional[AnswerCorrectnessExtractor],
    model_name: str,
    max_context_chars: int = 270000,
) -> dict:
    ctx = truncate_context(sample_data.get("retrieved_contexts", "") or "", max_context_chars)

    sample = {
        "user_input": sample_data.get("query_text", ""),
        "question": sample_data.get("query_text", ""),
        "response": sample_data.get("response_answer", ""),
        "answer": sample_data.get("response_answer", ""),
        "reference": sample_data.get("ground_truth", ""),
        "ground_truth": sample_data.get("ground_truth", ""),
        "retrieved_contexts": [ctx] if ctx else [],
        "contexts": [ctx] if ctx else [],
    }

    out = {
        "query_id": sample_data.get("query_id", ""),
        "query_text": sample["user_input"],
        "background_scenario": sample_data.get("background_scenario", ""),
        "ground_truth": sample["reference"],
        "response_answer": sample["response"],
        "retrieved_contexts": ctx,
        "model_evaluator": model_name,
    }

    # 1. Faithfulness
    if fe and ctx:
        try:
            score = await fe._ascore(sample)
            explanation, items = fe.extract_reasoning(fe.get_last_results())
            out["faithfulness_score"] = score
            out["faithfulness_reasoning"] = explanation
            out["faithfulness_details_json"] = json.dumps(items, ensure_ascii=False)
        except Exception as e:
            out["faithfulness_score"] = ""
            out["faithfulness_reasoning"] = f"ERROR: {e}"
            out["faithfulness_details_json"] = "[]"
    else:
        out["faithfulness_score"] = ""
        out["faithfulness_reasoning"] = "skipped (no context)"
        out["faithfulness_details_json"] = "[]"

    # 2. Context Recall
    if ce and ctx and sample["reference"]:
        try:
            score = await ce._ascore(sample)
            explanation, items = ce.extract_reasoning(ce.get_last_results())
            out["context_recall_score"] = score
            out["context_recall_reasoning"] = explanation
            out["context_recall_details_json"] = json.dumps(items, ensure_ascii=False)
        except Exception as e:
            out["context_recall_score"] = ""
            out["context_recall_reasoning"] = f"ERROR: {e}"
            out["context_recall_details_json"] = "[]"
    else:
        out["context_recall_score"] = ""
        out["context_recall_reasoning"] = "skipped (no context or reference)"
        out["context_recall_details_json"] = "[]"

    # 3. Answer Correctness
    if ae and sample["reference"]:
        try:
            score = await ae._ascore(sample)
            results = ae.get_last_results()
            explanation, details = ae.extract_reasoning(results)
            out["answer_correctness_score"] = score
            out["answer_correctness_f1"] = results.get("f1_score", "")
            out["answer_correctness_similarity"] = results.get("similarity_score", "")
            out["answer_correctness_reasoning"] = explanation
            out["answer_correctness_details_json"] = json.dumps(details, ensure_ascii=False)
        except Exception as e:
            out["answer_correctness_score"] = ""
            out["answer_correctness_f1"] = ""
            out["answer_correctness_similarity"] = ""
            out["answer_correctness_reasoning"] = f"ERROR: {e}"
            out["answer_correctness_details_json"] = "{}"
    else:
        out["answer_correctness_score"] = ""
        out["answer_correctness_f1"] = ""
        out["answer_correctness_similarity"] = ""
        out["answer_correctness_reasoning"] = "skipped (no reference)"
        out["answer_correctness_details_json"] = "{}"

    return out


class EvaluationRunner:
    """
    Kelas pengelola alur kerja evaluasi batch RAGAS dengan checkpointing dan audit log.
    """

    def __init__(
        self,
        backend: str,
        model_name: str,
        embedding_backend: str = "hf",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_ctx: int = 131072,
        metrics: t.List[str] = None,
        logger: t.Optional[logging.Logger] = None,
    ):
        self.backend = backend
        self.model_name = model_name
        self.metrics = metrics or ["faithfulness", "context_recall", "answer_correctness"]
        self.logger = logger or logging.getLogger("ragas_eval")

        self.logger.info(f"Menginisialisasi evaluator LLM '{model_name}' (backend: {backend})...")
        self.evaluator_llm = build_evaluator_llm(backend, model_name, num_ctx=num_ctx)
        self.embeddings = (
            build_evaluator_embeddings(embedding_backend, embedding_model)
            if "answer_correctness" in self.metrics
            else None
        )

        self.fe = FaithfulnessExtractor(llm_callable=self.evaluator_llm) if "faithfulness" in self.metrics else None
        self.ce = ContextRecallExtractor(llm_callable=self.evaluator_llm) if "context_recall" in self.metrics else None
        self.ae = (
            AnswerCorrectnessExtractor(llm_callable=self.evaluator_llm, evaluator_embeddings=self.embeddings)
            if "answer_correctness" in self.metrics
            else None
        )
        self.logger.info(f"Metrik aktif: {self.metrics}")

    async def run(
        self,
        samples: t.List[dict],
        output_csv_path: str,
        limit: int = 0,
        max_context_chars: int = 270000,
        metadata_extra: t.Optional[dict] = None,
    ):
        start_time = time.time()
        start_iso = datetime.now().isoformat()
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        existing_ids = set()
        if os.path.exists(output_csv_path):
            with open(output_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    qid = r.get("query_id") or r.get("query_text")
                    if qid:
                        existing_ids.add(qid)

        pending = []
        for s in samples:
            qid = s.get("query_id") or s.get("query_text")
            if qid not in existing_ids:
                pending.append(s)

        if limit > 0:
            pending = pending[:limit]

        self.logger.info(
            f"Total input: {len(samples)} | Sudah ada di CSV: {len(existing_ids)} | Akan diproses: {len(pending)}"
        )

        file_mode = "a" if existing_ids else "w"
        write_header = not existing_ids

        all_results = []
        with open(output_csv_path, file_mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            if write_header:
                writer.writeheader()

            for idx, sample in enumerate(tqdm(pending, desc=f"Evaluating ({self.model_name})")):
                item_start = time.time()
                res = await evaluate_single_sample(
                    sample_data=sample,
                    fe=self.fe,
                    ce=self.ce,
                    ae=self.ae,
                    model_name=self.model_name,
                    max_context_chars=max_context_chars,
                )
                writer.writerow(res)
                f.flush()
                all_results.append(res)
                item_duration = time.time() - item_start

                # Log rincian per baris ke file log dan terminal dengan output reasoning
                sample_log_block = format_sample_evaluation(
                    idx=idx + 1,
                    total=len(pending),
                    res=res,
                    duration=item_duration,
                )
                self.logger.info("\n" + sample_log_block)

        end_time = time.time()
        total_duration = end_time - start_time

        # Hitung statistik ringkas untuk run metadata
        def _avg(scores):
            valid = [float(s) for s in scores if s != "" and s is not None and not str(s).startswith("ERROR")]
            return sum(valid) / len(valid) if valid else None

        avg_f = _avg([r.get("faithfulness_score") for r in all_results])
        avg_cr = _avg([r.get("context_recall_score") for r in all_results])
        avg_ac = _avg([r.get("answer_correctness_score") for r in all_results])

        meta_info = {
            "experiment_type": "Tugas 2(b) LLM-as-a-judge RAGAS Evaluator",
            "model_evaluator": self.model_name,
            "backend": self.backend,
            "start_time": start_iso,
            "end_time": datetime.now().isoformat(),
            "duration_seconds": round(total_duration, 2),
            "output_csv": output_csv_path,
            "samples_processed": len(all_results),
            "total_samples": len(samples),
            "metrics_evaluated": self.metrics,
            "summary_scores": {
                "mean_faithfulness": round(avg_f, 4) if avg_f is not None else None,
                "mean_context_recall": round(avg_cr, 4) if avg_cr is not None else None,
                "mean_answer_correctness": round(avg_ac, 4) if avg_ac is not None else None,
            },
        }
        if metadata_extra:
            meta_info.update(metadata_extra)

        meta_json_path = f"{output_csv_path}.meta.json"
        with open(meta_json_path, "w", encoding="utf-8") as jf:
            json.dump(meta_info, jf, indent=2, ensure_ascii=False)

        c = AnsiColor
        self.logger.info(f"{c.CYAN}{'=' * 75}{c.RESET}")
        self.logger.info(f"{c.BOLD}{c.BRIGHT_GREEN}[RINGKASAN EKSEKUSI SELESAI]{c.RESET}")
        self.logger.info(f"Output CSV      : {output_csv_path}")
        self.logger.info(f"Metadata Bukti  : {meta_json_path}")
        self.logger.info(f"Durasi Total    : {total_duration:.2f} detik")
        self.logger.info(f"Rata-rata Skor  : {meta_info['summary_scores']}")
        self.logger.info(f"{c.CYAN}{'=' * 75}{c.RESET}")

        print(f"\n[SUKSES] Evaluasi selesai!")
        print(f"   - File Hasil    : {output_csv_path}")
        print(f"   - Metadata JSON : {meta_json_path}")
