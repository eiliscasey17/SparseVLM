#!/bin/bash

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
        echo "Auto-selected relatively free GPU ${CUDA_VISIBLE_DEVICES} (min_free_mem=${min_free_mem_mib} MiB, max_util=${max_util_pct}%)"
        return 0
    fi

    if [ -n "${fallback_gpu}" ]; then
        export CUDA_VISIBLE_DEVICES="${fallback_gpu}"
        echo "No GPU met the free-GPU thresholds; falling back to least-loaded GPU ${CUDA_VISIBLE_DEVICES}"
        return 0
    fi

    echo "No GPUs detected by nvidia-smi; leaving CUDA_VISIBLE_DEVICES unset"
}

EXPERIMENT="multiverse get attention"
ANSWERS_DIR="./playground/data/eval/multiverse/answers"
ANSWERS_FILE="$ANSWERS_DIR/${EXPERIMENT}.jsonl"
RESULTS_FILE="./playground/data/eval/multiverse/results.txt"

mkdir -p "$ANSWERS_DIR"

echo "Running MultiVerse evaluation..."

select_best_gpu

START_TIME=$(date +%s)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" python -m llava.eval.multiverse_experiment \
    --model-path liuhaotian/llava-v1.5-7b \
    --dataset-name passing2961/MultiVerse \
    --split train \
    --output-file results.jsonl \
    --conv-mode vicuna_v1 \
    --generate-responses \
    --answers-file "$ANSWERS_FILE"
END_TIME=$(date +%s)

ELAPSED=$((END_TIME - START_TIME))
echo "multiverse | ${EXPERIMENT} | ${ELAPSED}s" >> "$RESULTS_FILE"

echo "Done!"
