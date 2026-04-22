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
BASE_DIR="${BASE_DIR:-./playground/data/eval/mmbench/cluster_score_compare/${SPLIT}}"
LLM_RESULTS_DIR="${LLM_RESULTS_DIR:-${BASE_DIR}/llm_scores}"
ENCODER_RESULTS_DIR="${ENCODER_RESULTS_DIR:-${BASE_DIR}/encoder_scores}"
PLOT_FILE="${PLOT_FILE:-${BASE_DIR}/cluster_score_compare.png}"
PLOT_TITLE="${PLOT_TITLE:-MMBench Cluster Removal: LLM vs Encoder Scoring}"
TOKEN_SWEEP_SUMMARY="${TOKEN_SWEEP_SUMMARY:-./playground/data/eval/mmbench/attention_sweep/${SPLIT}/summary.tsv}"
ENCODER_TOKEN_SUMMARY="${ENCODER_TOKEN_SUMMARY:-./playground/data/eval/mmbench/encoder_sweep/${SPLIT}/summary.tsv}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
NUM_BEAMS="${NUM_BEAMS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
LANGUAGE="${LANGUAGE:-en}"
KMEANS_NUM_CLUSTERS="${KMEANS_NUM_CLUSTERS:-16}"
KMEANS_SPATIAL_WEIGHT="${KMEANS_SPATIAL_WEIGHT:-0.05}"
SEMANTIC_NUM_CLUSTERS="${SEMANTIC_NUM_CLUSTERS:-16}"
SEMANTIC_CLUSTER_PAGE_SIZE="${SEMANTIC_CLUSTER_PAGE_SIZE:-32}"
MIN_CLUSTER_QUESTIONS="${MIN_CLUSTER_QUESTIONS:-100}"

mkdir -p "$LLM_RESULTS_DIR" "$ENCODER_RESULTS_DIR"

select_free_gpus

run_sweep() {
    local scoring_source="$1"
    local results_dir="$2"

    CMD=(
        python -m llava.eval.model_vqa_mmbench_cluster_sweep
        --model-path "$MODEL_PATH"
        --question-file "$QUESTION_FILE"
        --results-dir "$results_dir"
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
        --scoring-source "$scoring_source"
    )

    if [[ -n "$MODEL_BASE" ]]; then
        CMD+=(--model-base "$MODEL_BASE")
    fi

    "${CMD[@]}"
}

echo "Running cluster sweep with LLM scores..."
run_sweep "llm" "$LLM_RESULTS_DIR"

echo "Running cluster sweep with encoder scores..."
run_sweep "encoder" "$ENCODER_RESULTS_DIR"

COMBINED_SUMMARY="${BASE_DIR}/combined_summary.tsv"
{
    head -n 1 "${LLM_RESULTS_DIR}/summary.tsv"
    tail -n +2 "${LLM_RESULTS_DIR}/summary.tsv"
    tail -n +2 "${ENCODER_RESULTS_DIR}/summary.tsv"
} > "$COMBINED_SUMMARY"

echo "Building comparison plot..."
python scripts/plot_mmbench_cluster_vs_token_sweep.py \
    --cluster-summary-file "$COMBINED_SUMMARY" \
    --token-summary-file "$TOKEN_SWEEP_SUMMARY" \
    --encoder-summary-file "$ENCODER_TOKEN_SUMMARY" \
    --output-file "$PLOT_FILE" \
    --title "$PLOT_TITLE" \
    --min-cluster-questions "$MIN_CLUSTER_QUESTIONS"

echo "Done."
echo "LLM summary: ${LLM_RESULTS_DIR}/summary.tsv"
echo "Encoder summary: ${ENCODER_RESULTS_DIR}/summary.tsv"
echo "Combined summary: $COMBINED_SUMMARY"
echo "Plot: $PLOT_FILE"
