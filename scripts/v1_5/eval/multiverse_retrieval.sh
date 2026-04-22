#!/bin/bash

set -e

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

MODEL_PATH="liuhaotian/llava-v1.5-7b"
DATASET_NAME="passing2961/MultiVerse"
SPLIT="train"
CONV_MODE="llava_v1"
SAMPLE_SIZE="20"

PRUNE_RATIO="0.2"
NUM_SINK_TOKENS="32"
NUM_CLUSTERS="16"
CLUSTERING_MODE="kmeans"
CLUSTER_PAGE_SIZE=""
TOPK_CLUSTERS="4"
RETRIEVAL_MODE="cosine_mean"
TOKEN_BUDGET=""
USE_PROJECTED_TOKENS="true"

PYTHON_SCRIPT="./llava/eval/multiverse_retrieval.py"

OUTPUT_DIR="./playground/data/eval/multiverse_retrieval"
OUTPUT_FILE="${OUTPUT_DIR}/retrieval_results.jsonl"
RESULTS_FILE="${OUTPUT_DIR}/runtime_log.txt"
RESULTS_JSONL="${OUTPUT_DIR}/results.jsonl"
RESULTS_TSV="${OUTPUT_DIR}/results.tsv"
EXPERIMENT="multiverse_retrieval_pr${PRUNE_RATIO}_sink${NUM_SINK_TOKENS}_c${NUM_CLUSTERS}_topk${TOPK_CLUSTERS}_${RETRIEVAL_MODE}_${CLUSTERING_MODE}"
if [ -n "${CLUSTER_PAGE_SIZE}" ]; then
    EXPERIMENT="${EXPERIMENT}_ps${CLUSTER_PAGE_SIZE}"
fi

mkdir -p "${OUTPUT_DIR}"

echo "Running VLM retrieval (sink + cluster)..."

select_best_gpu

CMD=(
    python "${PYTHON_SCRIPT}"
    --model-path "${MODEL_PATH}"
    --dataset-name "${DATASET_NAME}"
    --split "${SPLIT}"
    --output-file "${OUTPUT_FILE}"
    --conv-mode "${CONV_MODE}"
    --sample-size "${SAMPLE_SIZE}"
    --prune-ratio "${PRUNE_RATIO}"
    --num-sink-tokens "${NUM_SINK_TOKENS}"
    --num-clusters "${NUM_CLUSTERS}"
    --clustering-mode "${CLUSTERING_MODE}"
    --topk-clusters "${TOPK_CLUSTERS}"
    --retrieval-mode "${RETRIEVAL_MODE}"
)

if [ -n "${CLUSTER_PAGE_SIZE}" ]; then
    CMD+=(--cluster-page-size "${CLUSTER_PAGE_SIZE}")
fi

if [ -n "${TOKEN_BUDGET}" ]; then
    CMD+=(--token-budget "${TOKEN_BUDGET}")
fi

if [ "${USE_PROJECTED_TOKENS}" = "true" ]; then
    CMD+=(--use-projected-tokens-for-output)
fi

START_TIME=$(date +%s)

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" "${CMD[@]}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

TIMESTAMP="$(date -Iseconds)"

KEEP_STATS=$(python - <<PY
import json

selected = []
kept_pct = []

with open(${OUTPUT_FILE@Q}, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if "selected_vision_tokens" in item:
            selected.append(float(item["selected_vision_tokens"]))
        if "kept_vision_token_pct" in item:
            kept_pct.append(float(item["kept_vision_token_pct"]))

avg_selected = sum(selected) / len(selected) if selected else 0.0
avg_kept_pct = sum(kept_pct) / len(kept_pct) if kept_pct else 0.0
print(f"{avg_selected:.4f}|{avg_kept_pct:.4f}")
PY
)
IFS='|' read -r AVG_SELECTED_VISION_TOKENS AVG_KEPT_VISION_TOKEN_PCT <<< "${KEEP_STATS}"

if [ ! -f "${RESULTS_TSV}" ]; then
    printf "timestamp\tdataset\tsplit\texperiment\tmodel_path\tconv_mode\tsample_size\tprune_ratio\tnum_sink_tokens\tnum_clusters\tclustering_mode\tcluster_page_size\ttopk_clusters\tretrieval_mode\ttoken_budget\tuse_projected_tokens\tavg_selected_vision_tokens\tavg_kept_vision_token_pct\telapsed_sec\taccuracy\toutput_file\n" > "${RESULTS_TSV}"
fi

printf "multiverse_retrieval | experiment=%s | elapsed_sec=%s\n" "${EXPERIMENT}" "${ELAPSED}" >> "${RESULTS_FILE}"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${TIMESTAMP}" \
    "${DATASET_NAME}" \
    "${SPLIT}" \
    "${EXPERIMENT}" \
    "${MODEL_PATH}" \
    "${CONV_MODE}" \
    "${SAMPLE_SIZE}" \
    "${PRUNE_RATIO}" \
    "${NUM_SINK_TOKENS}" \
    "${NUM_CLUSTERS}" \
    "${CLUSTERING_MODE}" \
    "${CLUSTER_PAGE_SIZE}" \
    "${TOPK_CLUSTERS}" \
    "${RETRIEVAL_MODE}" \
    "${TOKEN_BUDGET}" \
    "${USE_PROJECTED_TOKENS}" \
    "${AVG_SELECTED_VISION_TOKENS}" \
    "${AVG_KEPT_VISION_TOKEN_PCT}" \
    "${ELAPSED}" \
    "" \
    "${OUTPUT_FILE}" >> "${RESULTS_TSV}"

python - <<PY
import json
from pathlib import Path

record = {
    "timestamp": ${TIMESTAMP@Q},
    "dataset": ${DATASET_NAME@Q},
    "split": ${SPLIT@Q},
    "experiment": ${EXPERIMENT@Q},
    "model_path": ${MODEL_PATH@Q},
    "conv_mode": ${CONV_MODE@Q},
    "sample_size": None if ${SAMPLE_SIZE@Q} == "all" else int(${SAMPLE_SIZE@Q}),
    "prune_ratio": float(${PRUNE_RATIO@Q}),
    "num_sink_tokens": int(${NUM_SINK_TOKENS@Q}),
    "num_clusters": int(${NUM_CLUSTERS@Q}),
    "clustering_mode": ${CLUSTERING_MODE@Q},
    "cluster_page_size": None if not ${CLUSTER_PAGE_SIZE@Q} else int(${CLUSTER_PAGE_SIZE@Q}),
    "topk_clusters": int(${TOPK_CLUSTERS@Q}),
    "retrieval_mode": ${RETRIEVAL_MODE@Q},
    "token_budget": None if not ${TOKEN_BUDGET@Q} else int(${TOKEN_BUDGET@Q}),
    "use_projected_tokens": ${USE_PROJECTED_TOKENS@Q}.lower() == "true",
    "avg_selected_vision_tokens": float(${AVG_SELECTED_VISION_TOKENS@Q}),
    "avg_kept_vision_token_pct": float(${AVG_KEPT_VISION_TOKEN_PCT@Q}),
    "elapsed_sec": int(${ELAPSED@Q}),
    "accuracy": None,
    "output_file": ${OUTPUT_FILE@Q},
}

path = Path(${RESULTS_JSONL@Q})
with path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record) + "\\n")
PY

cat <<EOF
Run Summary
dataset: ${DATASET_NAME}
split: ${SPLIT}
experiment: ${EXPERIMENT}
model_path: ${MODEL_PATH}
clustering_mode: ${CLUSTERING_MODE}
cluster_page_size: ${CLUSTER_PAGE_SIZE:-null}
prune_ratio: ${PRUNE_RATIO}
num_sink_tokens: ${NUM_SINK_TOKENS}
num_clusters: ${NUM_CLUSTERS}
topk_clusters: ${TOPK_CLUSTERS}
retrieval_mode: ${RETRIEVAL_MODE}
token_budget: ${TOKEN_BUDGET:-null}
use_projected_tokens: ${USE_PROJECTED_TOKENS}
sample_size: ${SAMPLE_SIZE}
avg_selected_vision_tokens: ${AVG_SELECTED_VISION_TOKENS}
avg_kept_vision_token_pct: ${AVG_KEPT_VISION_TOKEN_PCT}
elapsed_sec: ${ELAPSED}
accuracy: null
results_jsonl: ${RESULTS_JSONL}
results_tsv: ${RESULTS_TSV}
EOF

echo "Done!"
