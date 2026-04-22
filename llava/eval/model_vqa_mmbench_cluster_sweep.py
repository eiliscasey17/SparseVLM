import argparse
import json
import math
import os
import re
import time

import pandas as pd
import shortuuid
import torch
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    load_image_from_base64,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.model.vision_memory import cluster_tokens_kmeans, cluster_tokens_semantic
from llava.utils import disable_torch_init


ALL_OPTIONS = ["A", "B", "C", "D"]


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def is_none(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.lower() in {"nan", "none"}:
        return True
    return False


def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def build_question_text(row, options):
    question = row["question"]
    hint = row["hint"]
    if not is_none(hint):
        question = hint + "\n" + question
    for option_char, option in zip(ALL_OPTIONS[: len(options)], options):
        question = question + "\n" + option_char + ". " + option
    return question


def build_prompt(question_text, model, args):
    qs = question_text
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if args.single_pred_prompt:
        if args.lang == "cn":
            qs = qs + "\n请直接回答选项字母。"
        else:
            qs = qs + "\nAnswer with the option's letter from the given choices directly."

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def maybe_cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def extract_option_letter(text, valid_options):
    upper_text = text.strip().upper()
    matches = re.findall(r"\b([A-D])\b", upper_text)
    for match in matches:
        if match in valid_options:
            return match
    for char in upper_text:
        if char in valid_options:
            return char
    return upper_text


def infer_patch_grid(model):
    vision_tower = model.get_vision_tower()
    num_patches_per_side = getattr(vision_tower, "num_patches_per_side", None)
    if num_patches_per_side is None:
        return None
    return (num_patches_per_side, num_patches_per_side)


def compute_token_scores_and_patch_features(model, input_ids, image_tensor, image_size, scoring_source="llm"):
    prepare_attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    (
        _,
        position_ids,
        attention_mask,
        _,
        inputs_embeds,
        _,
        vision_token_metadata,
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        position_ids=None,
        attention_mask=prepare_attention_mask,
        past_key_values=None,
        labels=None,
        images=image_tensor.unsqueeze(0).half().cuda(),
        image_sizes=[image_size],
        agnostic_sparsity=0.0,
        aware_sparsity=0.0,
        return_vision_token_metadata=True,
    )

    batch_metadata = vision_token_metadata[0]
    image_token_ranges = batch_metadata["image_token_ranges"]
    if not image_token_ranges:
        empty_scores = torch.empty(0, dtype=torch.float32)
        empty_features = torch.empty(0, dtype=torch.float32)
        return empty_scores, empty_features, 0.0

    vision_tower = model.get_vision_tower()
    tower = vision_tower.vision_tower

    with torch.no_grad():
        vision_output = tower(
            image_tensor.unsqueeze(0).half().cuda(),
            output_attentions=True,
            return_dict=True,
        )
    raw_vision_features = vision_output.last_hidden_state.squeeze(0).detach().float()

    if scoring_source == "llm":
        maybe_cuda_synchronize()
        start_time = time.perf_counter()
        with torch.inference_mode():
            outputs = model.get_model()(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
        maybe_cuda_synchronize()
        ranking_elapsed_seconds = time.perf_counter() - start_time

        attentions = outputs.attentions[-1]
        seq_len = attentions.shape[-1]
        attention_device = attentions.device
        vision_positions = []
        text_query_mask = torch.ones(seq_len, dtype=torch.bool, device=attention_device)
        for start, end in image_token_ranges:
            if end > start:
                vision_positions.append(torch.arange(start, end, device=attention_device))
                text_query_mask[start:end] = False

        if not vision_positions or not torch.any(text_query_mask):
            empty_scores = torch.empty(0, dtype=torch.float32)
            empty_features = torch.empty(0, dtype=torch.float32)
            return empty_scores, empty_features, ranking_elapsed_seconds

        vision_positions = torch.cat(vision_positions, dim=0)
        vision_scores = attentions[:, :, text_query_mask, :][:, :, :, vision_positions]
        vision_scores = vision_scores.sum(dim=(1, 2)).squeeze(0).detach().float().cpu()
    elif scoring_source == "encoder":
        maybe_cuda_synchronize()
        start_time = time.perf_counter()
        cls_attention = vision_output.attentions[-1][:, :, 0, :]
        maybe_cuda_synchronize()
        ranking_elapsed_seconds = time.perf_counter() - start_time
        vision_scores = cls_attention.sum(dim=1).squeeze(0).detach().float().cpu()
    else:
        raise ValueError(f"Unknown scoring_source: {scoring_source}")

    if raw_vision_features.shape[0] != vision_scores.shape[0]:
        raise ValueError(
            "Vision feature/token score mismatch: "
            f"{raw_vision_features.shape[0]} features vs {vision_scores.shape[0]} scores."
        )

    if raw_vision_features.shape[0] <= 1:
        patch_features = torch.empty(0, raw_vision_features.shape[-1], dtype=torch.float32)
    else:
        patch_features = raw_vision_features[1:].cpu()

    return vision_scores, patch_features, ranking_elapsed_seconds


def build_cluster_runs(method_name, cluster_ids, patch_scores, full_vision_token_count):
    if patch_scores.numel() == 0 or cluster_ids.numel() == 0:
        return [
            {
                "method": method_name,
                "run_label": "baseline",
                "clusters_total": 0,
                "clusters_kept": 0,
                "kept_patch_tokens": max(0, full_vision_token_count - 1),
                "kept_vision_tokens": full_vision_token_count,
                "keep_mask": None,
            }
        ]

    unique_clusters = torch.unique(cluster_ids).tolist()
    cluster_infos = []
    for cluster_id in unique_clusters:
        patch_indices = torch.where(cluster_ids == cluster_id)[0]
        mean_score = patch_scores[patch_indices].mean().item()
        cluster_infos.append(
            {
                "cluster_id": int(cluster_id),
                "patch_indices": patch_indices,
                "mean_score": mean_score,
            }
        )

    cluster_infos.sort(key=lambda item: item["mean_score"])
    clusters_total = len(cluster_infos)

    runs = []
    for removed_count in range(0, clusters_total):
        kept_clusters = cluster_infos[removed_count:]
        keep_mask = torch.zeros(full_vision_token_count, dtype=torch.bool)
        keep_mask[0] = True
        kept_patch_tokens = 0
        for cluster in kept_clusters:
            original_patch_indices = cluster["patch_indices"] + 1
            keep_mask[original_patch_indices] = True
            kept_patch_tokens += int(cluster["patch_indices"].numel())

        runs.append(
            {
                "method": method_name,
                "run_label": "baseline" if removed_count == 0 else f"remove_{removed_count}_cluster",
                "clusters_total": clusters_total,
                "clusters_kept": len(kept_clusters),
                "kept_patch_tokens": kept_patch_tokens,
                "kept_vision_tokens": int(keep_mask.sum().item()),
                "keep_mask": None if removed_count == 0 else keep_mask,
            }
        )
    return runs


def run_generation(model, tokenizer, input_ids, image_tensor, image_size, args, vision_token_keep_mask=None):
    generate_kwargs = {
        "images": image_tensor.unsqueeze(0).half().cuda(),
        "image_sizes": [image_size],
        "do_sample": args.temperature > 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "return_dict_in_generate": True,
    }
    if vision_token_keep_mask is not None:
        generate_kwargs["vision_token_keep_masks"] = [vision_token_keep_mask.cuda()]

    maybe_cuda_synchronize()
    start_time = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            **generate_kwargs,
        )
    maybe_cuda_synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    sequences = output.sequences
    generated_sequences = sequences[:, input_ids.shape[1] :]
    decode_source = generated_sequences if generated_sequences.shape[1] > 0 else sequences
    output_text = tokenizer.batch_decode(decode_source, skip_special_tokens=True)[0].strip()
    generated_token_count = int(sequences.shape[-1] - input_ids.shape[1])
    return output_text, generated_token_count, elapsed_seconds


def init_summary():
    return {
        "count": 0,
        "correct": 0,
        "generation_seconds": 0.0,
        "generated_tokens": 0,
        "kept_tokens": 0,
        "clusters_kept": 0,
        "clusters_total": 0,
    }


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        attn_implementation="eager",
    )

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(
            f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}."
        )

    patch_grid = infer_patch_grid(model)

    os.makedirs(args.results_dir, exist_ok=True)
    predictions_path = os.path.join(args.results_dir, "predictions.jsonl")
    summary_path = os.path.join(args.results_dir, "summary.tsv")
    config_path = os.path.join(args.results_dir, "config.json")

    predictions_file = open(predictions_path, "w", encoding="utf-8")
    summaries = {}
    total_ranking_seconds = 0.0

    for _, row in tqdm(questions.iterrows(), total=len(questions)):
        idx = row["index"]
        options = get_options(row, ALL_OPTIONS)
        valid_option_letters = ALL_OPTIONS[: len(options)]
        question_text = build_question_text(row, options)
        prompt = build_prompt(question_text, model, args)
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        image = load_image_from_base64(row["image"]).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]

        vision_scores, patch_features, ranking_elapsed_seconds = compute_token_scores_and_patch_features(
            model,
            input_ids,
            image_tensor,
            image.size,
            scoring_source=args.scoring_source,
        )
        total_ranking_seconds += ranking_elapsed_seconds
        full_vision_token_count = int(vision_scores.shape[0])
        gt_answer = str(row["answer"]).strip().upper()

        if full_vision_token_count == 0:
            continue

        if patch_features.numel() == 0:
            patch_scores = torch.empty(0, dtype=torch.float32)
            kmeans_runs = build_cluster_runs(f"kmeans_cluster_{args.scoring_source}", torch.empty(0, dtype=torch.long), patch_scores, full_vision_token_count)
            semantic_runs = build_cluster_runs(f"semantic_page_cluster_{args.scoring_source}", torch.empty(0, dtype=torch.long), patch_scores, full_vision_token_count)
        else:
            patch_scores = vision_scores[1:]
            patch_features_cuda = patch_features.cuda()
            kmeans_cluster_ids = cluster_tokens_kmeans(
                patch_features_cuda,
                num_clusters=args.kmeans_num_clusters,
                patch_grid=patch_grid,
                original_indices=None,
                spatial_weight=args.kmeans_spatial_weight,
            ).cpu()
            semantic_cluster_ids = cluster_tokens_semantic(
                patch_features_cuda,
                num_clusters=args.semantic_num_clusters,
                cluster_page_size=args.semantic_cluster_page_size,
            ).cpu()

            kmeans_runs = build_cluster_runs(
                f"kmeans_cluster_{args.scoring_source}",
                kmeans_cluster_ids,
                patch_scores,
                full_vision_token_count,
            )
            semantic_runs = build_cluster_runs(
                f"semantic_page_cluster_{args.scoring_source}",
                semantic_cluster_ids,
                patch_scores,
                full_vision_token_count,
            )

        for run in kmeans_runs + semantic_runs:
            key = (
                run["method"],
                run["clusters_kept"],
                run["clusters_total"],
                run["kept_vision_tokens"],
            )
            if key not in summaries:
                summaries[key] = init_summary()

            output_text, generated_token_count, elapsed_seconds = run_generation(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                image_tensor=image_tensor,
                image_size=image.size,
                args=args,
                vision_token_keep_mask=run["keep_mask"],
            )

            pred_answer = extract_option_letter(output_text, valid_option_letters)
            is_correct = int(pred_answer == gt_answer)

            summary = summaries[key]
            summary["count"] += 1
            summary["correct"] += is_correct
            summary["generation_seconds"] += elapsed_seconds
            summary["generated_tokens"] += generated_token_count
            summary["kept_tokens"] += run["kept_vision_tokens"]
            summary["clusters_kept"] += run["clusters_kept"]
            summary["clusters_total"] += run["clusters_total"]

            predictions_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_name,
                        "method": run["method"],
                        "run_label": run["run_label"],
                        "clusters_total": run["clusters_total"],
                        "clusters_kept": run["clusters_kept"],
                        "full_vision_token_count": full_vision_token_count,
                        "kept_vision_token_count": run["kept_vision_tokens"],
                        "kept_patch_token_count": run["kept_patch_tokens"],
                        "ranking_elapsed_seconds": ranking_elapsed_seconds,
                        "generation_elapsed_seconds": elapsed_seconds,
                        "generated_tokens": generated_token_count,
                        "ground_truth": gt_answer,
                        "prediction_text": output_text,
                        "prediction_option": pred_answer,
                        "is_correct": bool(is_correct),
                    }
                )
                + "\n"
            )
            predictions_file.flush()

    predictions_file.close()

    rows = []
    for (method, clusters_kept, clusters_total, kept_vision_tokens), summary in summaries.items():
        accuracy = summary["correct"] / summary["count"] if summary["count"] else 0.0
        avg_kept_tokens = summary["kept_tokens"] / summary["count"] if summary["count"] else 0.0
        avg_generation_seconds = summary["generation_seconds"] / summary["count"] if summary["count"] else 0.0
        avg_generated_tokens = summary["generated_tokens"] / summary["count"] if summary["count"] else 0.0
        avg_clusters_kept = summary["clusters_kept"] / summary["count"] if summary["count"] else 0.0
        avg_clusters_total = summary["clusters_total"] / summary["count"] if summary["count"] else 0.0
        rows.append(
            {
                "method": method,
                "clusters_kept": clusters_kept,
                "clusters_total": clusters_total,
                "questions": summary["count"],
                "correct": summary["correct"],
                "accuracy": accuracy,
                "avg_kept_vision_tokens": avg_kept_tokens,
                "total_generation_seconds": summary["generation_seconds"],
                "avg_generation_seconds": avg_generation_seconds,
                "avg_generated_tokens": avg_generated_tokens,
                "avg_clusters_kept": avg_clusters_kept,
                "avg_clusters_total": avg_clusters_total,
            }
        )

    rows.sort(key=lambda item: (item["method"], -item["avg_kept_vision_tokens"]))

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(
            "\t".join(
                [
                    "method",
                    "clusters_kept",
                    "clusters_total",
                    "questions",
                    "correct",
                    "accuracy",
                    "avg_kept_vision_tokens",
                    "total_generation_seconds",
                    "avg_generation_seconds",
                    "avg_generated_tokens",
                    "avg_clusters_kept",
                    "avg_clusters_total",
                ]
            )
            + "\n"
        )
        for row in rows:
            summary_file.write(
                "\t".join(
                    [
                        row["method"],
                        str(row["clusters_kept"]),
                        str(row["clusters_total"]),
                        str(row["questions"]),
                        str(row["correct"]),
                        f"{row['accuracy']:.6f}",
                        f"{row['avg_kept_vision_tokens']:.2f}",
                        f"{row['total_generation_seconds']:.6f}",
                        f"{row['avg_generation_seconds']:.6f}",
                        f"{row['avg_generated_tokens']:.2f}",
                        f"{row['avg_clusters_kept']:.2f}",
                        f"{row['avg_clusters_total']:.2f}",
                    ]
                )
                + "\n"
            )

    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(
            {
                "model_path": args.model_path,
                "model_base": args.model_base,
                "question_file": args.question_file,
                "results_dir": args.results_dir,
                "conv_mode": args.conv_mode,
                "kmeans_num_clusters": args.kmeans_num_clusters,
                "kmeans_spatial_weight": args.kmeans_spatial_weight,
                "semantic_num_clusters": args.semantic_num_clusters,
                "semantic_cluster_page_size": args.semantic_cluster_page_size,
                "scoring_source": args.scoring_source,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens,
                "total_ranking_seconds": total_ranking_seconds,
                "average_ranking_seconds": total_ranking_seconds / len(questions) if len(questions) else 0.0,
            },
            config_file,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--question-file",
        type=str,
        default="playground/data/eval/mmbench/mmbench_dev_20230712.tsv",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="playground/data/eval/mmbench/cluster_sweep",
    )
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--kmeans-num-clusters", type=int, default=16)
    parser.add_argument("--kmeans-spatial-weight", type=float, default=0.05)
    parser.add_argument("--semantic-num-clusters", type=int, default=16)
    parser.add_argument("--semantic-cluster-page-size", type=int, default=32)
    parser.add_argument("--scoring-source", type=str, default="llm", choices=["llm", "encoder"])
    args = parser.parse_args()

    eval_model(args)
