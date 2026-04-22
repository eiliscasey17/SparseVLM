#!/bin/bash

# Dataset split
SPLIT="mmbench_dev_20230712"

# Sparsity level (you can change this per run)
SPARSE="99"

# Experiment name (you can change this per run)
EXPERIMENT="sparsity_cluster_batchpruning_${SPARSE}"

# Base answers directory
ANSWERS_DIR="./playground/data/eval/mmbench/answers/$SPLIT"
ANSWERS_UPLOAD="./playground/data/eval/mmbench/answers_upload/${SPLIT}_${EXPERIMENT}"

# Input question file
QUESTION_FILE="./playground/data/eval/mmbench/$SPLIT.tsv"

RESULTS_FILE="./playground/data/eval/mmbench/results.txt"

# Make directories
mkdir -p "$ANSWERS_DIR"
mkdir -p "$ANSWERS_UPLOAD"

# Dynamic answers file path: answers + experiment
ANSWERS_FILE="$ANSWERS_DIR/${EXPERIMENT}.jsonl"

# Run evaluation and time it
echo "Running LLaVA evaluation..."

START_TIME=$(date +%s)
python -m llava.eval.model_vqa_mmbench \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file $QUESTION_FILE \
    --answers-file $ANSWERS_FILE \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Convert for submission
echo "Converting results for submission..."


python scripts/convert_mmbench_for_submission.py \
    --annotation-file $QUESTION_FILE \
    --result-dir $ANSWERS_DIR \
    --upload-dir $ANSWERS_UPLOAD \
    --experiment llava-v1.5-13b

echo "Computing accuracy..."
ACCURACY_OUTPUT=$(python playground/data/eval/mmbench/compute_accuracy.py \
    --pred-file "$ANSWERS_FILE" \
    --gt-file "$QUESTION_FILE" \
    --experiment-name "$EXPERIMENT")
echo "mmbench | $ACCURACY_OUTPUT | ${ELAPSED}s" >> "$RESULTS_FILE"

echo "Done!"