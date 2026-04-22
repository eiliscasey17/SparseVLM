#!/bin/bash

select_best_gpu_count() {
    local gpu_count="${1:-1}"

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
    local rows=""

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
        local eligible=0
        if [ "${free_mem}" -ge "${min_free_mem_mib}" ] && [ "${util}" -le "${max_util_pct}" ]; then
            eligible=1
        fi
        local score=$((eligible * 1000000000 - util * 100000 - mem_used))
        rows+="${score},${index}"$'\n'
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)

    if [ -z "${rows}" ]; then
        echo "No GPUs detected by nvidia-smi; leaving CUDA_VISIBLE_DEVICES unset"
        return 0
    fi

    local selected
    selected=$(printf "%s" "${rows}" | sort -t, -k1,1nr -k2,2n | head -n "${gpu_count}" | cut -d, -f2 | paste -sd, -)
    export CUDA_VISIBLE_DEVICES="${selected}"
    echo "Auto-selected GPU(s) ${CUDA_VISIBLE_DEVICES} for model ${MODEL_PATH}"
}

desired_gpu_count() {
    if [ -n "${GPU_COUNT:-}" ]; then
        echo "${GPU_COUNT}"
        return 0
    fi

    case "${MODEL_PATH,,}" in
        *13b*) echo "4" ;;
        *) echo "1" ;;
    esac
}

EXPERIMENT="multiverse_baseline"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-13b}"
ANSWERS_DIR="./playground/data/eval/multiverse/answers"
ANSWERS_FILE="$ANSWERS_DIR/${EXPERIMENT}.jsonl"
RESULTS_FILE="./playground/data/eval/multiverse/results.txt"

mkdir -p "$ANSWERS_DIR"

echo "Running MultiVerse evaluation..."

select_best_gpu_count "$(desired_gpu_count)"

START_TIME=$(date +%s)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" python -m llava.eval.model_vqa_multiverse \
    --model-path "${MODEL_PATH}" \
    --dataset-name passing2961/MultiVerse \
    --split train \
    --answers-file "$ANSWERS_FILE" \
    --temperature 0 \
    --conv-mode vicuna_v1
END_TIME=$(date +%s)

ELAPSED=$((END_TIME - START_TIME))
echo "multiverse | ${EXPERIMENT} | ${ELAPSED}s" >> "$RESULTS_FILE"

echo "Done!"
