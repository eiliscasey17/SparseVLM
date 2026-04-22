import argparse
import ast
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
from PIL import Image
import math

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_list_field(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if not isinstance(x, str):
        return [x]
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except Exception:
        return [x]


def load_image(image_file):
    return Image.open(image_file).convert("RGB")


def resolve_image_paths(image_paths, image_folder):
    resolved = []
    for p in image_paths:
        if os.path.isabs(p):
            resolved.append(p)
        else:
            resolved.append(os.path.join(image_folder, p))
    return resolved


def eval_model(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w", encoding="utf-8")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'Auto switching conv mode to {args.conv_mode}')

    for _, row in tqdm(questions.iterrows(), total=len(questions)):
        idx = row["index"]
        question_list = parse_list_field(row["question"])
        answer_list = parse_list_field(row["answer"])
        image_path_list = parse_list_field(row["image_path"])
        subset = row.get("set", "")

        image_paths = resolve_image_paths(image_path_list, args.image_folder)

        history = []
        predictions = []
        pics_number = 0

        for question in question_list:
            question = str(question)

            new_images = question.count("<ImageHere>")
            if new_images > 0:
                pics_number += new_images

            current_images = image_paths[:pics_number]

            qs = question.replace("<ImageHere>", "<image-placeholder>").strip()
            if model.config.mm_use_im_start_end:
                qs = qs.replace(
                    "<image-placeholder>",
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
            else:
                qs = qs.replace("<image-placeholder>", DEFAULT_IMAGE_TOKEN)

            conv = conv_templates[args.conv_mode].copy()

            for user_q, assistant_a in history:
                conv.append_message(conv.roles[0], user_q)
                conv.append_message(conv.roles[1], assistant_a)

            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).cuda()

            pil_images = [load_image(p) for p in current_images]
            image_sizes = [img.size for img in pil_images]
            image_tensor = process_images(pil_images, image_processor, model.config)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.half().cuda(),
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            predictions.append(outputs)
            history.append((qs, outputs))

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "questions": question_list,
            "predictions": predictions,
            "answers": answer_list,
            "image_path": image_path_list,
            "set": subset,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }, ensure_ascii=False) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    eval_model(args)