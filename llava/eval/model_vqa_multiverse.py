import argparse
import os
import json
import math
import shortuuid
import torch
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def build_first_user_prompt(sample, user_text, model):
    intro = []
    if sample.get("character"):
        intro.append(f"Character profile: {sample['character']}")
    if sample.get("scenario"):
        intro.append(f"Scenario: {sample['scenario']}")
    if sample.get("goal"):
        intro.append(f"Goal: {sample['goal']}")

    intro_text = "\n".join(intro).strip()

    if intro_text:
        text = intro_text + "\n\nUser: " + user_text
    else:
        text = user_text

    if model.config.mm_use_im_start_end:
        return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
    else:
        return DEFAULT_IMAGE_TOKEN + "\n" + text


def eval_model(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    ds = load_dataset(
        args.dataset_name,
        split=args.split,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    if args.sample_size != "all":
        ds = ds.select(range(min(int(args.sample_size), len(ds))))

    samples = list(ds)
    samples = get_chunk(samples, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, "w") as ans_file:
        for sample in tqdm(samples):
            idx = sample["index"]

            image = sample["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")

            image_tensor = process_images([image], image_processor, model.config)[0]

            speakers = sample["conversation"]["speaker"]
            utterances = sample["conversation"]["utterance"]
            checklist = sample["conversation"].get("checklist", [])

            conv = conv_templates[args.conv_mode].copy()

            assistant_turn_counter = 0

            for turn_idx, (speaker, utterance) in enumerate(zip(speakers, utterances)):
                if speaker.upper() == "USER":
                    if len(conv.messages) == 0:
                        qs = build_first_user_prompt(sample, utterance, model)
                    else:
                        qs = utterance
                    conv.append_message(conv.roles[0], qs)

                elif speaker.upper() == "ASSISTANT":
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    input_ids = tokenizer_image_token(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
                    ).unsqueeze(0).cuda()

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=image_tensor.unsqueeze(0).half().cuda(),
                            image_sizes=[image.size],
                            agnostic_sparsity=args.agnostic_sparsity,
                            aware_sparsity=args.aware_sparsity,
                            do_sample=args.temperature > 0,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            num_beams=args.num_beams,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                            return_dict_in_generate=False
                        )

                    output_ids = output_ids[:, input_ids.shape[1]:]
                    outputs = tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True
                    )[0].strip()

                    # Save prediction for this assistant turn
                    turn_checklist = checklist[turn_idx] if turn_idx < len(checklist) else {}

                    ans_file.write(json.dumps({
                        "question_id": f"{idx}_{assistant_turn_counter}",
                        "sample_id": idx,
                        "turn_id": turn_idx,
                        "assistant_turn_id": assistant_turn_counter,
                        "character": sample.get("character", ""),
                        "scenario": sample.get("scenario", ""),
                        "goal": sample.get("goal", ""),
                        "prompt": conv.messages[-2][1] if len(conv.messages) >= 2 else "",
                        "user_prompt": conv.messages[-2][1] if len(conv.messages) >= 2 else "",
                        "text": outputs,
                        "prediction": outputs,
                        "reference": utterance,
                        "checklist": turn_checklist,
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_name,
                        "metadata": {}
                    }) + "\n")
                    ans_file.flush()

                    # Replace the placeholder None with model output so dialogue history continues
                    conv.messages[-1][-1] = outputs
                    assistant_turn_counter += 1

    print(f"Saved results to {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default="passing2961/MultiVerse")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sample-size", type=str, default="all")
    parser.add_argument("--agnostic-sparsity", type=float, default=0.0)
    parser.add_argument("--aware-sparsity", type=float, default=0.0)
    args = parser.parse_args()

    eval_model(args)
