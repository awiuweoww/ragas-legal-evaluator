#!/bin/bash
# ==============================================================================
# Copyright (c) 2026 Badan Riset dan Inovasi Nasional (BRIN)
# THIS SOFTWARE SOURCE CODE AND ANY EXECUTABLE DERIVED THEREOF ARE PROPRIETARY
# TO BRIN, AS APPLICABLE, AND SHALL NOT BE USED IN ANY WAY
# OTHER THAN BEFOREHAND AGREED ON BY BRIN, NOR BE REPRODUCED
# OR DISCLOSED TO THIRD PARTIES WITHOUT PRIOR WRITTEN AUTHORIZATION BY
# BRIN, AS APPLICABLE.
#
# Author      : Chandra
# Version     : 0.1.0
# Created     : 13-10-2026
# Description : Script runner evaluasi RAGAS via OpenAI API backend (GPT-4o-mini, GPT-4o).
#
# Changelog:
# - 0.1.0 (13-10-2026): Inisialisasi runner script OpenAI API.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_PATH="${1:-"${ROOT_DIR}/data/qa_rag/perikanan_naive_qa.csv"}"
MODEL_NAME="${2:-"gpt-4o-mini"}"
OUTPUT_PATH="${3:-"${ROOT_DIR}/data/results/eval_openai_${MODEL_NAME}.csv"}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    if [ -f "${ROOT_DIR}/.env" ]; then
        export $(grep -v '^#' "${ROOT_DIR}/.env" | xargs -d '\n')
    fi
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY belum diset. Isi file .env atau export OPENAI_API_KEY=..."
    exit 1
fi

echo "=========================================================="
echo "▶ Menjalankan Evaluasi RAGAS (Backend: OpenAI)"
echo "  Dataset : ${DATASET_PATH}"
echo "  Model   : ${MODEL_NAME}"
echo "  Output  : ${OUTPUT_PATH}"
echo "  Log     : Otomatis tersimpan di ${ROOT_DIR}/logs/"
echo "=========================================================="

python3 "${SCRIPT_DIR}/run_eval.py" \
    --input "${DATASET_PATH}" \
    --output "${OUTPUT_PATH}" \
    --backend "openai" \
    --model "${MODEL_NAME}" \
    --embedding_backend "hf"
