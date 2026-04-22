#!/bin/bash

set -euo pipefail

SPLIT="${SPLIT:-mmbench_dev_20230712}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-13b}"
MODEL_BASE="${MODEL_BASE:-}"
QUESTION_FILE="${QUESTION_FILE:-./playground/data/eval/mmbench/${SPLIT}.tsv}"
RESULTS_DIR="${RESULTS_DIR:-./playground/data/eval/mmbench/attention_sweep/${SPLIT}}"
SUMMARY_FILE="${SUMMARY_FILE:-${RESULTS_DIR}/summary.tsv}"
PLOT_FILE="${PLOT_FILE:-${RESULTS_DIR}/accuracy_time_vs_retained_tokens.svg}"
PLOT_TITLE="${PLOT_TITLE:-Retained Vision Tokens vs Accuracy and Runtime}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
NUM_BEAMS="${NUM_BEAMS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
RETAINED_RATIO_STEP="${RETAINED_RATIO_STEP:-}"
RETAINED_RATIOS="${RETAINED_RATIOS:-}"
RETENTION_TARGETS="${RETENTION_TARGETS:-}"
LANGUAGE="${LANGUAGE:-en}"

mkdir -p "$RESULTS_DIR"

echo "Running attention sweep..."

SWEEP_CMD=(
    python -m llava.eval.model_vqa_mmbench_attention_sweep
    --model-path "$MODEL_PATH"
    --question-file "$QUESTION_FILE"
    --results-dir "$RESULTS_DIR"
    --conv-mode "$CONV_MODE"
    --temperature "$TEMPERATURE"
    --num_beams "$NUM_BEAMS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --lang "$LANGUAGE"
    --single-pred-prompt
)

if [[ -n "$MODEL_BASE" ]]; then
    SWEEP_CMD+=(--model-base "$MODEL_BASE")
fi

if [[ -n "$RETAINED_RATIOS" ]]; then
    SWEEP_CMD+=(--retained-ratios "$RETAINED_RATIOS")
fi

if [[ -n "$RETENTION_TARGETS" ]]; then
    SWEEP_CMD+=(--retention-targets "$RETENTION_TARGETS")
fi

if [[ -n "$RETAINED_RATIO_STEP" ]]; then
    SWEEP_CMD+=(--retained-ratio-step "$RETAINED_RATIO_STEP")
fi

"${SWEEP_CMD[@]}"

echo "Building plot..."

python scripts/plot_mmbench_attention_sweep.py \
    --summary-file "$SUMMARY_FILE" \
    --output-file "$PLOT_FILE" \
    --title "$PLOT_TITLE"

echo "Done."
echo "Summary: $SUMMARY_FILE"
echo "Plot: $PLOT_FILE"
