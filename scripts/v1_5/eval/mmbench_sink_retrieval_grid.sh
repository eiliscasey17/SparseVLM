#!/bin/bash

set -euo pipefail

select_best_gpu() {
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
    local best_gpu=""
    local best_tiebreak=""
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
            local tiebreak=$((util * 100000 + mem_used))
            if [ -z "${best_gpu}" ] || [ "${tiebreak}" -lt "${best_tiebreak}" ]; then
                best_gpu="${index}"
                best_tiebreak="${tiebreak}"
            fi
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)

    if [ -n "${best_gpu}" ]; then
        export CUDA_VISIBLE_DEVICES="${best_gpu}"
        echo "Auto-selected relatively free GPU ${CUDA_VISIBLE_DEVICES}"
        return 0
    fi

    if [ -n "${fallback_gpu}" ]; then
        export CUDA_VISIBLE_DEVICES="${fallback_gpu}"
        echo "No GPU met the thresholds; falling back to GPU ${CUDA_VISIBLE_DEVICES}"
        return 0
    fi

    echo "No GPUs detected by nvidia-smi; leaving CUDA_VISIBLE_DEVICES unset"
}

PYTHON_BIN="${PYTHON_BIN:-/home/caseye2/miniconda3/envs/llava_projects/bin/python}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
SPLIT="${SPLIT:-mmbench_dev_20230712}"
QUESTION_FILE="${QUESTION_FILE:-./playground/data/eval/mmbench/${SPLIT}.tsv}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-./playground/data/eval/mmbench/sink_retrieval_grid}"

VISION_TOKEN_COUNT="${VISION_TOKEN_COUNT:-576}"
PRUNE_RATIO="${PRUNE_RATIO:-0.5}"
SINK_PERCENTAGES="${SINK_PERCENTAGES:-20,40,60,80,100}"
RETRIEVAL_PERCENTAGES="${RETRIEVAL_PERCENTAGES:-10,20,30,40,50,60,70,80,90,100}"

NUM_CLUSTERS="${NUM_CLUSTERS:-16}"
CLUSTERING_MODE="${CLUSTERING_MODE:-kmeans}"
CLUSTER_PAGE_SIZE="${CLUSTER_PAGE_SIZE:-4096}"
TOPK_CLUSTERS="${TOPK_CLUSTERS:-4}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-cosine_mean}"
USE_PROJECTED_TOKENS="${USE_PROJECTED_TOKENS:-true}"
PLOT_ONLY="${PLOT_ONLY:-false}"
RERUN_EXISTING="${RERUN_EXISTING:-false}"

select_best_gpu

CMD=(
    "${PYTHON_BIN}"
    "scripts/v1_5/eval/mmbench_sink_retrieval_grid.py"
    --model-path "${MODEL_PATH}"
    --split "${SPLIT}"
    --question-file "${QUESTION_FILE}"
    --conv-mode "${CONV_MODE}"
    --output-dir "${OUTPUT_DIR}"
    --vision-token-count "${VISION_TOKEN_COUNT}"
    --prune-ratio "${PRUNE_RATIO}"
    --sink-percentages "${SINK_PERCENTAGES}"
    --retrieval-percentages "${RETRIEVAL_PERCENTAGES}"
    --num-clusters "${NUM_CLUSTERS}"
    --clustering-mode "${CLUSTERING_MODE}"
    --topk-clusters "${TOPK_CLUSTERS}"
    --retrieval-mode "${RETRIEVAL_MODE}"
)

if [ -n "${MODEL_BASE}" ]; then
    CMD+=(--model-base "${MODEL_BASE}")
fi

if [ -n "${CLUSTER_PAGE_SIZE}" ]; then
    CMD+=(--cluster-page-size "${CLUSTER_PAGE_SIZE}")
fi

if [ "${USE_PROJECTED_TOKENS}" = "true" ]; then
    CMD+=(--use-projected-tokens-for-output)
fi

if [ "${PLOT_ONLY}" = "true" ]; then
    CMD+=(--plot-only)
fi

if [ "${RERUN_EXISTING}" = "true" ]; then
    CMD+=(--rerun-existing)
fi

echo "Running MMBench sink/retrieval grid sweep..."
echo "Python: ${PYTHON_BIN}"
echo "Model: ${MODEL_PATH}"
echo "Split: ${SPLIT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Prune ratio: ${PRUNE_RATIO}"
echo "Sink percentages: ${SINK_PERCENTAGES}"
echo "Retrieval percentages: ${RETRIEVAL_PERCENTAGES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

START_TIME=$(date +%s)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" "${CMD[@]}"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "Sweep finished in ${ELAPSED}s"
echo "Manifest: ${OUTPUT_DIR}/manifest.json"
echo "Results JSON: ${OUTPUT_DIR}/mmbench_sink_retrieval_grid_results.json"
echo "Results TSV: ${OUTPUT_DIR}/mmbench_sink_retrieval_grid_results.tsv"
echo "Runtime heatmap: ${OUTPUT_DIR}/mmbench_sink_retrieval_grid_runtime.png"
echo "Accuracy heatmap: ${OUTPUT_DIR}/mmbench_sink_retrieval_grid_accuracy.png"
echo "Configured-x plot: ${OUTPUT_DIR}/mmbench_sink_retrieval_grid_configuredx.png"
echo "Actual-x plot: ${OUTPUT_DIR}/mmbench_sink_retrieval_grid_actualx.png"
