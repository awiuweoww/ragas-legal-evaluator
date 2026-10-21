#!/usr/bin/env python3
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
Description : CLI Runner utama evaluasi RAGAS multi-backend (Ollama & OpenAI)
              dengan pencatatan log otomatis (audit trail) dan metadata bukti run.

Changelog:
- 0.1.0 (13-10-2026): Implementasi CLI fleksibel pendeteksi kolom otomatis dan eksekusi batch terstruktur.
- 0.1.1 (13-10-2026): Penambahan auto-logging ke berkas log bertanggal di folder logs/ untuk bukti riset.
"""

import argparse
import asyncio
import csv
import logging
import os
import sys
import typing as t
from datetime import datetime
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

# Hindari error field larger than field limit untuk konteks GraphRAG yang besar
_max_field_size = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max_field_size)
        break
    except OverflowError:
        _max_field_size //= 10

load_dotenv(os.path.join(ROOT_DIR, ".env"))

from src.evaluators.runner import EvaluationRunner
from src.utils.logger import setup_logger, AnsiColor


def detect_columns(fieldnames: list) -> dict:
    candidates = {
        "query_text": ["Question", "question", "Pertanyaan", "query", "Query"],
        "ground_truth": ["Gold Answer", "Gold_Answer", "Ground_Truth", "ground_truth", "Answer", "reference", "Reference"],
        "response_answer": ["LLM_Response", "minirag", "naiverag", "Prediction", "response", "Answer_Generated", "Answer"],
        "retrieved_contexts": ["Context", "Contexts", "retrieved_contexts", "context", "Referensi_Hukum", "Reference_Provisions"],
        "background_scenario": ["Background_Scenario", "Background Scenario", "Scenario", "Skenario"],
        "query_id": ["ID", "query_id", "PairIndex", "Filename"],
    }
    mapping = {}
    lower_map = {f.lower(): f for f in fieldnames}

    for target, cand_list in candidates.items():
        matched = None
        for c in cand_list:
            if c in fieldnames:
                matched = c
                break
            elif c.lower() in lower_map:
                matched = lower_map[c.lower()]
                break
        mapping[target] = matched

    return mapping


def load_dataset(input_path: str) -> list:
    with open(input_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        col_map = detect_columns(fields)

        samples = []
        for idx, row in enumerate(reader):
            qid = row.get(col_map["query_id"]) if col_map["query_id"] else f"query_{idx+1}"
            samples.append({
                "query_id": qid,
                "query_text": row.get(col_map["query_text"], "") if col_map["query_text"] else "",
                "ground_truth": row.get(col_map["ground_truth"], "") if col_map["ground_truth"] else "",
                "response_answer": row.get(col_map["response_answer"], "") if col_map["response_answer"] else "",
                "retrieved_contexts": row.get(col_map["retrieved_contexts"], "") if col_map["retrieved_contexts"] else "",
                "background_scenario": row.get(col_map["background_scenario"], "") if col_map["background_scenario"] else "",
            })
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Jalankan Evaluasi RAGAS Multi-Model dengan Output Reasoning & Auto-Logging")
    parser.add_argument("--input", required=True, help="Path ke input dataset (CSV)")
    parser.add_argument("--output", required=True, help="Path output hasil evaluasi (CSV)")
    parser.add_argument("--backend", choices=["ollama", "openai", "vllm"], default="openai")
    parser.add_argument("--model", required=True, help="Nama LLM Evaluator (e.g. gpt-4o-mini, gpt-5.4, gemma3:12b, qwen2.5:14b)")
    parser.add_argument("--embedding_backend", choices=["hf", "openai"], default="hf")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--num_ctx", type=int, default=131072, help="Context length untuk Ollama")
    parser.add_argument("--limit", type=int, default=0, help="Batasi hanya mengevaluasi N baris pertama (0=semua)")
    parser.add_argument("--metrics", default="faithfulness,context_recall,answer_correctness", help="Daftar metrik dipisah koma")
    parser.add_argument("--log_file", default=None, help="Path kustom file log (opsional)")
    return parser.parse_args()


def main():
    args = parse_args()
    logger, log_file_path = setup_logger(args.model, args.log_file)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    samples = load_dataset(args.input)

    c = AnsiColor
    logger.info(f"{c.CYAN}{'=' * 75}{c.RESET}")
    logger.info(f"{c.BOLD}{c.BRIGHT_CYAN}[START] MEMULAI RUN EVALUASI RAGAS (TUGAS 2b){c.RESET}")
    logger.info(f"Tanggal & Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input Dataset  : {args.input} ({len(samples)} baris)")
    logger.info(f"Evaluator LLM  : {c.BOLD}{args.model}{c.RESET} (Backend: {args.backend})")
    logger.info(f"Metrik Aktif   : {metrics}")
    logger.info(f"Output File    : {args.output}")
    logger.info(f"Berkas Log     : {log_file_path}")
    logger.info(f"{c.CYAN}{'=' * 75}{c.RESET}")

    runner = EvaluationRunner(
        backend=args.backend,
        model_name=args.model,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        num_ctx=args.num_ctx,
        metrics=metrics,
        logger=logger,
    )

    asyncio.run(
        runner.run(
            samples=samples,
            output_csv_path=args.output,
            limit=args.limit,
            metadata_extra={
                "input_dataset": args.input,
                "log_file": log_file_path,
            },
        )
    )


if __name__ == "__main__":
    main()
