#!/bin/bash

QUESTION_FILE="./playground/data/eval/MMDU/MMDU.tsv"
IMAGE_FOLDER="./playground/data/eval/MMDU/images"
ANSWERS_DIR="./playground/data/eval/MMDU/answers"
ANSWERS_FILE="${ANSWERS_DIR}/llava-v1.5-13b.jsonl"

mkdir -p "$ANSWERS_DIR"

START_TIME=$(date +%s)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m llava.eval.model_vqa_mmdu \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$ANSWERS_FILE" \
    --conv-mode llava_v1 \
    --temperature 0 \
    --num-beams 1 \
    --max-new-tokens 1024
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "mmbench | MMDU | llava-v1.5-13b | ${ELAPSED}s" >> "./playground/data/eval/MMDU/results.txt"

echo "Done!"