import argparse
import json
import math
import os

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
from llava.model.builder_sparsity import load_pretrained_model
from llava.model.vision_retriever import VisionRetrievalConfig
from llava.utils import disable_torch_init


ALL_OPTIONS = ["A", "B", "C", "D"]


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


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


def infer_patch_grid(model):
    vt = model.get_vision_tower()
    nps = getattr(vt, "num_patches_per_side", None)
    if nps is None:
        return None
    return (nps, nps)


def build_mmbench_prompt(row, options, model, args):
    question = row["question"]
    hint = row["hint"]
    if not is_none(hint):
        question = hint + "\n" + question

    for option_char, option in zip(ALL_OPTIONS[:len(options)], options):
        question += "\n" + option_char + ". " + option

    cur_prompt = question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question

    if args.single_pred_prompt:
        if args.lang == "cn":
            qs += "\n请直接回答选项字母。"
        else:
            qs += "\nAnswer with the option's letter from the given choices directly."

    return qs, cur_prompt


def sample_next_token(logits, temperature=0.0, top_p=None):
    if temperature is None or temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_p is not None and 0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_idx = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, next_idx)

    return torch.multinomial(probs, num_samples=1)


def generate_with_retrieval(
    model,
    tokenizer,
    input_ids,
    vision_memory,
    retrieval_cfg,
    max_new_tokens,
    temperature=0.0,
    top_p=None,
):
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    eos_token_id = tokenizer.eos_token_id
    generated = []
    past_key_values = None
    current_input_ids = input_ids
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                images=None,
                vision_memory=vision_memory,
                vision_retrieval_config=retrieval_cfg,
            )

        next_token = sample_next_token(
            outputs.logits[:, -1, :],
            temperature=temperature,
            top_p=top_p,
        )
        generated.append(next_token)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        past_key_values = outputs.past_key_values
        current_input_ids = next_token
        cached_seq_len = past_key_values[0][0].shape[-2]
        attention_mask = torch.ones(
            (1, cached_seq_len + 1),
            dtype=torch.long,
            device=input_ids.device,
        )

    if not generated:
        return ""

    output_ids = torch.cat(generated, dim=1)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def build_retrieval_stats(vision_memory, retrieval_info):
    total_vision_tokens = int(vision_memory.full_tokens.size(0))
    kept_after_prune_tokens = int(vision_memory.kept_original_indices.numel())
    available_non_sink_tokens = int(vision_memory.nonsink_original_indices.numel())
    selected_original_indices = retrieval_info.get("selected_original_indices", [])
    sink_original_indices = retrieval_info.get("sink_original_indices", [])
    selected_vision_tokens = len(selected_original_indices)
    sink_tokens_kept = len(sink_original_indices)
    kept_vision_token_pct = 0.0
    if total_vision_tokens > 0:
        kept_vision_token_pct = 100.0 * selected_vision_tokens / total_vision_tokens

    return {
        "total_vision_tokens": total_vision_tokens,
        "kept_after_prune_tokens": kept_after_prune_tokens,
        "available_non_sink_tokens": available_non_sink_tokens,
        "selected_vision_tokens": selected_vision_tokens,
        "sink_tokens_kept": sink_tokens_kept,
        "retrieved_non_sink_tokens": max(0, selected_vision_tokens - sink_tokens_kept),
        "kept_vision_token_pct": kept_vision_token_pct,
    }


def eval_model(args):
    disable_torch_init()

    if args.num_beams != 1:
        raise ValueError("Retrieval MMBench eval currently supports only --num-beams 1.")

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

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(
            f"It seems that this is a plain model, but it is not using a mmtag prompt, "
            f"auto switching to {args.conv_mode}."
        )

    retrieval_cfg = VisionRetrievalConfig(
        prune_ratio=args.prune_ratio,
        num_sink_tokens=args.num_sink_tokens,
        num_clusters=args.num_clusters,
        clustering_mode=args.clustering_mode,
        cluster_page_size=args.cluster_page_size,
        topk_clusters=args.topk_clusters,
        retrieval_mode=args.retrieval_mode,
        token_budget=args.token_budget,
        use_projected_tokens_for_output=args.use_projected_tokens_for_output,
    )
    patch_grid = infer_patch_grid(model)

    with open(answers_file, "w") as ans_file:
        for _, row in tqdm(questions.iterrows(), total=len(questions)):
            question_id = int(row["index"])
            base_options = get_options(row, ALL_OPTIONS)
            cur_option_char = ALL_OPTIONS[:len(base_options)]
            num_rounds = len(base_options) if args.all_rounds else 1
            options = list(base_options)

            image = load_image_from_base64(row["image"]).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)[0]
            image_input = image_tensor.unsqueeze(0).half().cuda()

            vision_memory = model.build_single_image_vision_memory(
                images=image_input,
                vision_retrieval_config=retrieval_cfg,
                patch_grid=patch_grid,
                metadata={"question_id": question_id},
            )
            if args.use_projected_tokens_for_output:
                vision_memory = model.cache_projected_tokens_in_memory(vision_memory)

            for round_idx in range(num_rounds):
                qs, cur_prompt = build_mmbench_prompt(row, options, model, args)

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(
                    prompt,
                    tokenizer,
                    IMAGE_TOKEN_INDEX,
                    return_tensors="pt",
                ).unsqueeze(0).cuda()

                _, retrieval_info = model.retrieve_projected_tokens_for_query(
                    vision_memory=vision_memory,
                    input_ids=input_ids,
                    image_token_index=IMAGE_TOKEN_INDEX,
                    vision_retrieval_config=retrieval_cfg,
                )
                retrieval_stats = build_retrieval_stats(vision_memory, retrieval_info)

                outputs = generate_with_retrieval(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    vision_memory=vision_memory,
                    retrieval_cfg=retrieval_cfg,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                ans_file.write(json.dumps({
                    "question_id": question_id,
                    "round_id": round_idx,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "options": options,
                    "option_char": cur_option_char,
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_name,
                    "metadata": {
                        "retrieval_mode": args.retrieval_mode,
                        "prune_ratio": args.prune_ratio,
                        "num_sink_tokens": args.num_sink_tokens,
                        "num_clusters": args.num_clusters,
                        "clustering_mode": args.clustering_mode,
                        "cluster_page_size": args.cluster_page_size,
                        "topk_clusters": args.topk_clusters,
                        "token_budget": args.token_budget,
                        "use_projected_tokens_for_output": args.use_projected_tokens_for_output,
                        **retrieval_stats,
                    },
                }) + "\n")
                ans_file.flush()

                options = options[1:] + options[:1]
                cur_option_char = cur_option_char[1:] + cur_option_char[:1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--max-new-tokens", type=int, default=16)

    parser.add_argument("--prune-ratio", type=float, default=0.2)
    parser.add_argument("--num-sink-tokens", type=int, default=32)
    parser.add_argument("--num-clusters", type=int, default=16)
    parser.add_argument("--clustering-mode", type=str, default="kmeans", choices=["kmeans", "semantic"])
    parser.add_argument("--cluster-page-size", type=int, default=None)
    parser.add_argument("--topk-clusters", type=int, default=4)
    parser.add_argument("--retrieval-mode", type=str, default="cosine_mean")
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--use-projected-tokens-for-output", action="store_true")

    eval_model(parser.parse_args())
