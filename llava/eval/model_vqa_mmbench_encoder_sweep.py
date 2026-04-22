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
from llava.utils import disable_torch_init


ALL_OPTIONS = ["A", "B", "C", "D"]
DEFAULT_RETENTION_TARGETS = [
    "1.00",
    "0.90",
    "0.80",
    "0.70",
    "0.60",
    "0.50",
    "0.40",
    "0.30",
    "0.20",
    "0.15",
    "0.10",
    "0.08",
    "0.05",
    "0.03",
    "0.01",
    "1tok",
]


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


def build_keep_mask(scores, retained_ratio=None, keep_count=None):
    total_tokens = scores.shape[0]
    if keep_count is None:
        if retained_ratio is None:
            raise ValueError("Either retained_ratio or keep_count must be provided.")
        if retained_ratio <= 0:
            keep_count = 0
        elif retained_ratio >= 1:
            keep_count = total_tokens
        else:
            keep_count = min(total_tokens, math.ceil(total_tokens * retained_ratio))
    else:
        keep_count = max(0, min(total_tokens, int(keep_count)))

    keep_mask = torch.zeros(total_tokens, dtype=torch.bool)
    if keep_count > 0:
        sorted_indices = torch.argsort(scores, descending=True)
        keep_mask[sorted_indices[:keep_count]] = True
    return keep_mask, keep_count


def parse_retention_target(spec):
    normalized = spec.strip().lower()
    if not normalized:
        raise ValueError("Empty retention target.")

    if normalized.endswith("%"):
        ratio = float(normalized[:-1]) / 100.0
        return {"label": f"{ratio:.4f}", "target_type": "ratio", "target_value": ratio}

    if normalized.endswith("tok") or normalized.endswith("token") or normalized.endswith("tokens"):
        digits = "".join(ch for ch in normalized if ch.isdigit())
        if not digits:
            raise ValueError(f"Could not parse token count from retention target '{spec}'.")
        return {"label": f"{int(digits)}tok", "target_type": "count", "target_value": int(digits)}

    ratio = float(normalized)
    if ratio > 1.0:
        ratio /= 100.0
    return {"label": f"{ratio:.4f}", "target_type": "ratio", "target_value": ratio}


def get_retention_targets(args):
    if args.retention_targets:
        return [parse_retention_target(item) for item in args.retention_targets.split(",") if item.strip()]
    return [parse_retention_target(item) for item in DEFAULT_RETENTION_TARGETS]


def compute_encoder_token_scores(model, image_tensor):
    vision_tower = model.get_vision_tower()
    tower = vision_tower.vision_tower

    maybe_cuda_synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        output = tower(
            image_tensor.unsqueeze(0).half().cuda(),
            output_attentions=True,
            return_dict=True,
        )
    maybe_cuda_synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    attentions = output.attentions[-1]
    cls_attention = attentions[:, :, 0, :]
    token_scores = cls_attention.sum(dim=1).squeeze(0).detach().float().cpu()
    return token_scores, elapsed_seconds


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
        output = model.generate(input_ids, **generate_kwargs)
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
        "full_tokens": 0,
        "effective_ratio_sum": 0.0,
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

    retention_targets = get_retention_targets(args)
    target_labels = [target["label"] for target in retention_targets]
    summaries = {label: init_summary() for label in target_labels}

    os.makedirs(args.results_dir, exist_ok=True)
    predictions_path = os.path.join(args.results_dir, "predictions.jsonl")
    summary_path = os.path.join(args.results_dir, "summary.tsv")
    config_path = os.path.join(args.results_dir, "config.json")

    total_ranking_seconds = 0.0
    predictions_file = open(predictions_path, "w", encoding="utf-8")

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

        encoder_scores, ranking_elapsed_seconds = compute_encoder_token_scores(model, image_tensor)
        total_ranking_seconds += ranking_elapsed_seconds

        full_vision_token_count = int(encoder_scores.shape[0])
        gt_answer = str(row["answer"]).strip().upper()

        for target in retention_targets:
            target_label = target["label"]
            if target["target_type"] == "ratio" and target["target_value"] >= 1.0:
                keep_mask = None
                kept_token_count = full_vision_token_count
            elif target["target_type"] == "ratio":
                keep_mask, kept_token_count = build_keep_mask(
                    encoder_scores,
                    retained_ratio=target["target_value"],
                )
            else:
                keep_mask, kept_token_count = build_keep_mask(
                    encoder_scores,
                    keep_count=target["target_value"],
                )

            effective_retained_ratio = (
                kept_token_count / full_vision_token_count if full_vision_token_count > 0 else 0.0
            )

            output_text, generated_token_count, elapsed_seconds = run_generation(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                image_tensor=image_tensor,
                image_size=image.size,
                args=args,
                vision_token_keep_mask=keep_mask,
            )

            pred_answer = extract_option_letter(output_text, valid_option_letters)
            is_correct = int(pred_answer == gt_answer)

            summaries[target_label]["count"] += 1
            summaries[target_label]["correct"] += is_correct
            summaries[target_label]["generation_seconds"] += elapsed_seconds
            summaries[target_label]["generated_tokens"] += generated_token_count
            summaries[target_label]["kept_tokens"] += kept_token_count
            summaries[target_label]["full_tokens"] += full_vision_token_count
            summaries[target_label]["effective_ratio_sum"] += effective_retained_ratio

            predictions_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_name,
                        "retention_target": target_label,
                        "target_type": target["target_type"],
                        "target_value": target["target_value"],
                        "retained_ratio": effective_retained_ratio,
                        "full_vision_token_count": full_vision_token_count,
                        "kept_vision_token_count": kept_token_count,
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

    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(
            "\t".join(
                [
                    "retention_target",
                    "target_type",
                    "target_value",
                    "retained_ratio",
                    "questions",
                    "correct",
                    "accuracy",
                    "avg_kept_vision_tokens",
                    "avg_full_vision_tokens",
                    "total_generation_seconds",
                    "avg_generation_seconds",
                    "avg_generated_tokens",
                ]
            )
            + "\n"
        )
        for target in retention_targets:
            target_label = target["label"]
            summary = summaries[target_label]
            accuracy = summary["correct"] / summary["count"] if summary["count"] else 0.0
            avg_effective_ratio = summary["effective_ratio_sum"] / summary["count"] if summary["count"] else 0.0
            avg_kept_tokens = summary["kept_tokens"] / summary["count"] if summary["count"] else 0.0
            avg_full_tokens = summary["full_tokens"] / summary["count"] if summary["count"] else 0.0
            avg_generation_seconds = summary["generation_seconds"] / summary["count"] if summary["count"] else 0.0
            avg_generated_tokens = summary["generated_tokens"] / summary["count"] if summary["count"] else 0.0
            summary_file.write(
                "\t".join(
                    [
                        target_label,
                        target["target_type"],
                        str(target["target_value"]),
                        f"{avg_effective_ratio:.6f}",
                        str(summary["count"]),
                        str(summary["correct"]),
                        f"{accuracy:.6f}",
                        f"{avg_kept_tokens:.2f}",
                        f"{avg_full_tokens:.2f}",
                        f"{summary['generation_seconds']:.6f}",
                        f"{avg_generation_seconds:.6f}",
                        f"{avg_generated_tokens:.2f}",
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
                "retention_targets": retention_targets,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens,
                "conv_mode": args.conv_mode,
                "single_pred_prompt": args.single_pred_prompt,
                "num_questions": len(questions),
                "total_ranking_seconds": total_ranking_seconds,
                "average_ranking_seconds": total_ranking_seconds / len(questions) if len(questions) else 0.0,
                "ranking_source": "vision_encoder_last_layer_cls_attention",
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
        default="playground/data/eval/mmbench/encoder_sweep",
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
    parser.add_argument("--retention-targets", type=str, default="")
    args = parser.parse_args()

    eval_model(args)
