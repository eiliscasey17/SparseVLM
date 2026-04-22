#!/bin/bash
 
set -e
 
MODEL_PATH="liuhaotian/llava-v1.5-7b"
DATASET_NAME="passing2961/MultiVerse"
SPLIT="train"
CONV_MODE="vicuna_v1"


PYTHON_SCRIPT="./llava/eval/multiverse.py"
 
OUTPUT_DIR="./playground/data/eval/multiverse"
OUTPUT_FILE="${OUTPUT_DIR}/sparsevila_keep_tokens_results_67_75agnostic.jsonl"
RESULTS_FILE="${OUTPUT_DIR}/runtime_log.txt"
 
mkdir -p "${OUTPUT_DIR}"
 
echo "Running SparseVILA-style keep-token analysis..."
 
START_TIME=$(date +%s)
 
CUDA_VISIBLE_DEVICES=0 python "${PYTHON_SCRIPT}" \
    --model-path "${MODEL_PATH}" \
    --dataset-name "${DATASET_NAME}" \
    --split "${SPLIT}" \
    --output-file "${OUTPUT_DIR}" \
    --conv-mode "${CONV_MODE}" \
    --num-chunks 1 \
    --chunk-idx 0 \
    --target-layers -1 \
    --aware-sparsity .67 \
    --agnostic-sparsity .75\
    --sample-size 20
 
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
 
echo "SparseVILA keep-token analysis | ${ELAPSED}s" >> "${RESULTS_FILE}"
echo "Done!"
