#!/bin/bash

set -euo pipefail

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

MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7Can you make the b}"
DATASET_NAME="${DATASET_NAME:-passing2961/MultiVerse}"
SPLIT="${SPLIT:-train}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
SAMPLE_SIZE="${SAMPLE_SIZE:-all}"
GPT_MODEL="${GPT_MODEL:-gpt-4.1-mini}"
RUN_NAME="${RUN_NAME:-multiverse_baseline}"
OUTPUT_DIR="${OUTPUT_DIR:-./playground/data/eval/multiverse}"

ANSWERS_DIR="${OUTPUT_DIR}/answers"
REVIEWS_DIR="${OUTPUT_DIR}/reviews"
SUMMARY_DIR="${OUTPUT_DIR}/summaries"

ANSWERS_FILE="${ANSWERS_DIR}/${RUN_NAME}.jsonl"
REVIEWS_FILE="${REVIEWS_DIR}/${RUN_NAME}_${GPT_MODEL}.jsonl"
SUMMARY_FILE="${SUMMARY_DIR}/${RUN_NAME}_${GPT_MODEL}.json"
RESULTS_FILE="${OUTPUT_DIR}/results.txt"

mkdir -p "${ANSWERS_DIR}" "${REVIEWS_DIR}" "${SUMMARY_DIR}"

select_best_gpu_count "$(desired_gpu_count)"

echo "Running MultiVerse generation..."
START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" python -m llava.eval.model_vqa_multiverse \
    --model-path "${MODEL_PATH}" \
    --dataset-name "${DATASET_NAME}" \
    --split "${SPLIT}" \
    --answers-file "${ANSWERS_FILE}" \
    --temperature 0 \
    --conv-mode "${CONV_MODE}" \
    --sample-size "${SAMPLE_SIZE}"

MID_TIME=$(date +%s)
echo "Running GPT evaluation with ${GPT_MODEL}..."

python -m llava.eval.eval_multiverse_gpt \
    --answers-file "${ANSWERS_FILE}" \
    --output-file "${REVIEWS_FILE}" \
    --summary-file "${SUMMARY_FILE}" \
    --gpt-model "${GPT_MODEL}"

END_TIME=$(date +%s)
GEN_ELAPSED=$((MID_TIME - START_TIME))
EVAL_ELAPSED=$((END_TIME - MID_TIME))
TOTAL_ELAPSED=$((END_TIME - START_TIME))

printf "multiverse_pipeline | run=%s | gpt_model=%s | generation_sec=%s | evaluation_sec=%s | total_sec=%s\n" \
    "${RUN_NAME}" "${GPT_MODEL}" "${GEN_ELAPSED}" "${EVAL_ELAPSED}" "${TOTAL_ELAPSED}" >> "${RESULTS_FILE}"

echo "Answers: ${ANSWERS_FILE}"
echo "Reviews: ${REVIEWS_FILE}"
echo "Summary: ${SUMMARY_FILE}"
