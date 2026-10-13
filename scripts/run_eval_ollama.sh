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
# Description : Script runner evaluasi RAGAS via Ollama backend (Gemma, Qwen).
#
# Changelog:
# - 0.1.0 (13-10-2026): Inisialisasi runner script Ollama.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Argumen: 1=Dataset Path, 2=Model Name, 3=Output Path
DATASET_PATH="${1:-"${ROOT_DIR}/data/qa_rag/perikanan_naive_qa.csv"}"
MODEL_NAME="${2:-"gemma3:12b"}"
OUTPUT_PATH="${3:-"${ROOT_DIR}/data/results/eval_ollama_${MODEL_NAME//:/_}.csv"}"

export OLLAMA_HOST="${OLLAMA_HOST:-"http://127.0.0.1:11434"}"

echo "=========================================================="
echo "▶ Menjalankan Evaluasi RAGAS (Backend: Ollama)"
echo "  Dataset : ${DATASET_PATH}"
echo "  Model   : ${MODEL_NAME}"
echo "  Output  : ${OUTPUT_PATH}"
echo "  Log     : Otomatis tersimpan di ${ROOT_DIR}/logs/"
echo "=========================================================="

python3 "${SCRIPT_DIR}/run_eval.py" \
    --input "${DATASET_PATH}" \
    --output "${OUTPUT_PATH}" \
    --backend "ollama" \
    --model "${MODEL_NAME}" \
    --num_ctx 131072 \
    --embedding_backend "hf"
