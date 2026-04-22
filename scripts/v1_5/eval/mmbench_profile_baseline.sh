#!/bin/bash

SPLIT="mmbench_dev_20230712"
MODEL_PATH="liuhaotian/llava-v1.5-13b"
EXPERIMENT="llava-v1.5-13b-profile"

QUESTION_FILE="./playground/data/eval/mmbench/${SPLIT}.tsv"
ANSWERS_DIR="./playground/data/eval/mmbench/answers/${SPLIT}"
METRICS_DIR="./playground/data/eval/mmbench/metrics/${SPLIT}"
UPLOAD_DIR="./playground/data/eval/mmbench/answers_upload/${SPLIT}_${EXPERIMENT}"

ANSWERS_FILE="${ANSWERS_DIR}/${EXPERIMENT}.jsonl"
METRICS_FILE="${METRICS_DIR}/${EXPERIMENT}.jsonl"

mkdir -p "${ANSWERS_DIR}"
mkdir -p "${METRICS_DIR}"
mkdir -p "${UPLOAD_DIR}"

python -m llava.eval.model_vqa_mmbench_profile \
    --model-path "${MODEL_PATH}" \
    --question-file "${QUESTION_FILE}" \
    --answers-file "${ANSWERS_FILE}" \
    --metrics-file "${METRICS_FILE}" \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_mmbench_for_submission.py \
    --annotation-file "${QUESTION_FILE}" \
    --result-dir "${ANSWERS_DIR}" \
    --upload-dir "${UPLOAD_DIR}" \
    --experiment "${EXPERIMENT}"
