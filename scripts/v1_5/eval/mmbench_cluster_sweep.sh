#!/bin/bash

set -euo pipefail

select_free_gpus() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "Using user-specified CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        return 0
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found; leaving CUDA_VISIBLE_DEVICES unset"
        return 0
    fi

    local candidates="${GPU_CANDIDATES:-}"
    local min_free_mem_mib="${GPU_MIN_FREE_MEM_MIB:-10000}"
    local max_util_pct="${GPU_MAX_UTIL_PCT:-30}"
    local free_gpus=()
    local fallback_gpu=""
    local fallback_score=""

    while IFS=',' read -r raw_index raw_mem_used raw_mem_total raw_util; do
        local index="${raw_index//[[:space:]]/}"
        local mem_used="${raw_mem_used//[[:space:]]/}"
        local mem_total="${raw_mem_total//[[:space:]]/}"
        local util="${raw_util//[[:space:]]/}"

        if [ -n "${candidates}" ]; then
            case ",${candidates}," in
                *,"${index}",*) ;;
                *) continue ;;
            esac
        fi

        local free_mem=$((mem_total - mem_used))
        local score=$((mem_used * 1000 + util * 100))

        if [ -z "${fallback_gpu}" ] || [ "${score}" -lt "${fallback_score}" ]; then
            fallback_gpu="${index}"
            fallback_score="${score}"
        fi

        if [ "${free_mem}" -ge "${min_free_mem_mib}" ] && [ "${util}" -le "${max_util_pct}" ]; then
            free_gpus+=("${index}")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)

    if [ "${#free_gpus[@]}" -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${free_gpus[*]}")"
        echo "Auto-selected relatively free GPUs ${CUDA_VISIBLE_DEVICES} (min_free_mem=${min_free_mem_mib} MiB, max_util=${max_util_pct}%)"
        return 0
    fi

    if [ -n "${fallback_gpu}" ]; then
        export CUDA_VISIBLE_DEVICES="${fallback_gpu}"
        echo "No GPU met the free-GPU thresholds; falling back to least-loaded GPU ${CUDA_VISIBLE_DEVICES}"
        return 0
    fi

    echo "No GPUs detected by nvidia-smi; leaving CUDA_VISIBLE_DEVICES unset"
}

SPLIT="${SPLIT:-mmbench_dev_20230712}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-13b}"
MODEL_BASE="${MODEL_BASE:-}"
QUESTION_FILE="${QUESTION_FILE:-./playground/data/eval/mmbench/${SPLIT}.tsv}"
RESULTS_DIR="${RESULTS_DIR:-./playground/data/eval/mmbench/cluster_sweep/${SPLIT}}"
SUMMARY_FILE="${SUMMARY_FILE:-${RESULTS_DIR}/summary.tsv}"
PLOT_FILE="${PLOT_FILE:-${RESULTS_DIR}/cluster_vs_token_accuracy.png}"
PLOT_TITLE="${PLOT_TITLE:-MMBench Accuracy vs Kept Vision Tokens}"
TOKEN_SWEEP_SUMMARY="${TOKEN_SWEEP_SUMMARY:-./playground/data/eval/mmbench/attention_sweep/${SPLIT}/summary.tsv}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
NUM_BEAMS="${NUM_BEAMS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
LANGUAGE="${LANGUAGE:-en}"
KMEANS_NUM_CLUSTERS="${KMEANS_NUM_CLUSTERS:-16}"
KMEANS_SPATIAL_WEIGHT="${KMEANS_SPATIAL_WEIGHT:-0.05}"
SEMANTIC_NUM_CLUSTERS="${SEMANTIC_NUM_CLUSTERS:-16}"
SEMANTIC_CLUSTER_PAGE_SIZE="${SEMANTIC_CLUSTER_PAGE_SIZE:-32}"

mkdir -p "$RESULTS_DIR"

select_free_gpus

echo "Running cluster sweep..."

SWEEP_CMD=(
    python -m llava.eval.model_vqa_mmbench_cluster_sweep
    --model-path "$MODEL_PATH"
    --question-file "$QUESTION_FILE"
    --results-dir "$RESULTS_DIR"
    --conv-mode "$CONV_MODE"
    --temperature "$TEMPERATURE"
    --num_beams "$NUM_BEAMS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --lang "$LANGUAGE"
    --single-pred-prompt
    --kmeans-num-clusters "$KMEANS_NUM_CLUSTERS"
    --kmeans-spatial-weight "$KMEANS_SPATIAL_WEIGHT"
    --semantic-num-clusters "$SEMANTIC_NUM_CLUSTERS"
    --semantic-cluster-page-size "$SEMANTIC_CLUSTER_PAGE_SIZE"
)

if [[ -n "$MODEL_BASE" ]]; then
    SWEEP_CMD+=(--model-base "$MODEL_BASE")
fi

"${SWEEP_CMD[@]}"

echo "Building comparison plot..."

python scripts/plot_mmbench_cluster_vs_token_sweep.py \
    --cluster-summary-file "$SUMMARY_FILE" \
    --token-summary-file "$TOKEN_SWEEP_SUMMARY" \
    --output-file "$PLOT_FILE" \
    --title "$PLOT_TITLE"

echo "Done."
echo "Summary: $SUMMARY_FILE"
echo "Plot: $PLOT_FILE"
