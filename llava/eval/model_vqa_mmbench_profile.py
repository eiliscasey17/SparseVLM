import argparse
import json
import math
import os
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


def build_text_query(row, options):
    question = row["question"]
    hint = row["hint"]
    if not is_none(hint):
        question = hint + "\n" + question
    for option_char, option in zip(ALL_OPTIONS[: len(options)], options):
        question = question + "\n" + option_char + ". " + option
    return question


def maybe_cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
    )

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    metrics_file = os.path.expanduser(args.metrics_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

    ans_file = open(answers_file, "w")
    metrics_out = open(metrics_file, "w")

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(
            f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}."
        )

    for _, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, ALL_OPTIONS)
        cur_option_char = ALL_OPTIONS[: len(options)]
        num_rounds = len(options) if args.all_rounds else 1

        for round_idx in range(num_rounds):
            idx = row["index"]
            text_query = build_text_query(row, options)
            image = load_image_from_base64(row["image"])

            qs = text_query
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
            prompt = conv.get_prompt()

            text_query_token_count = len(tokenizer(text_query, add_special_tokens=False).input_ids)
            prompt_token_count = len(tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX))
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            image_tensor = process_images([image], image_processor, model.config)[0]

            maybe_cuda_synchronize()
            start_time = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
            maybe_cuda_synchronize()
            elapsed_seconds = time.perf_counter() - start_time

            sequences = output.sequences
            output_text = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip()
            generated_token_count = sequences.shape[-1] - input_ids.shape[-1]

            ans_id = shortuuid.uuid()
            ans_record = {
                "question_id": idx,
                "round_id": round_idx,
                "prompt": text_query,
                "text": output_text,
                "options": options,
                "option_char": cur_option_char,
                "answer_id": ans_id,
                "model_id": model_name,
                "metadata": {},
            }
            ans_file.write(json.dumps(ans_record) + "\n")
            ans_file.flush()

            metrics_record = {
                "question_id": idx,
                "round_id": round_idx,
                "model_id": model_name,
                "text_query_tokens": text_query_token_count,
                "prompt_tokens": prompt_token_count,
                "generated_tokens": int(generated_token_count),
                "elapsed_seconds": elapsed_seconds,
                "image_width": image.size[0],
                "image_height": image.size[1],
            }
            metrics_out.write(json.dumps(metrics_record) + "\n")
            metrics_out.flush()

            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]

    ans_file.close()
    metrics_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--metrics-file", type=str, default="metrics.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    eval_model(args)
