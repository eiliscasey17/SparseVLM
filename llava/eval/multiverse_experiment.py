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
    '''intro = []
    if sample.get("character"):
        intro.append(f"Character profile: {sample['character']}")
    if sample.get("scenario"):
        intro.append(f"Scenario: {sample['scenario']}")
    if sample.get("goal"):
        intro.append(f"Goal: {sample['goal']}")

    intro_text = "\n".join(intro).strip()
    text = (intro_text + "\n\nUser: " + user_text) if intro_text else user_text

    if model.config.mm_use_im_start_end:
        return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
    else:
        return DEFAULT_IMAGE_TOKEN + "\n" + text'''
    
    text = user_text
    if model.config.mm_use_im_start_end:
        return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
    else:
        return DEFAULT_IMAGE_TOKEN + "\n" + text


def summarize_text_to_vision(attentions, text_idx, vision_idx):
    results = []
    for layer_idx, attn in enumerate(attentions):
        # attn: [B, H, Q, K]
        a = attn[0]  # [H, Q, K]
        block = a[:, text_idx, :][:, :, vision_idx]   # [H, T_text, T_vision]

        layer_result = {
            "layer": layer_idx,
            "mean_text_to_vision": block.mean().item(),
            "per_vision_token_mean": block.mean(dim=(0, 1)).detach().float().cpu().tolist(),
            "per_head_mean": block.mean(dim=(1, 2)).detach().float().cpu().tolist(),
        }
        results.append(layer_result)
    return results


def generate_assistant_reply(args, model, tokenizer, image_input, image_size, input_ids):
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_input,
            image_sizes=[image_size],
            agnostic_sparsity=args.agnostic_sparsity,
            aware_sparsity=args.aware_sparsity,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    output_ids = output_ids[:, input_ids.shape[1]:]
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def eval_model(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        attn_implementation="eager",   # important
    )

    ds = load_dataset(
        args.dataset_name,
        split=args.split,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    if args.sample_size != "all":
        ds = ds.select(range(min(int(args.sample_size), len(ds))))
    
    samples = get_chunk(list(ds), args.num_chunks, args.chunk_idx)

    output_dir = os.path.dirname(args.output_file)
    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    answers_file = None
    ans_f = None
    if args.generate_responses:
        answers_file = os.path.expanduser(args.answers_file)
        answers_dir = os.path.dirname(answers_file)
        if answers_dir:
            os.makedirs(answers_dir, exist_ok=True)
        ans_f = open(answers_file, "w", encoding="utf-8")

    with open(args.output_file, "w") as f:
        for sample in tqdm(samples):
            idx = sample["index"]

            image = sample["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")

            image_tensor = process_images([image], image_processor, model.config)[0]
            image_input = image_tensor.unsqueeze(0).half().cuda()

            speakers = sample["conversation"]["speaker"]
            utterances = sample["conversation"]["utterance"]

            conv = conv_templates[args.conv_mode].copy()
            assistant_turn_counter = 0

            for turn_idx, (speaker, utterance) in enumerate(zip(speakers, utterances)):

                if speaker.upper() == "USER":
                    if len(conv.messages) == 0:
                        qs = build_first_user_prompt(sample, utterance, model)
                    else:
                        qs = utterance
                    conv.append_message(conv.roles[0], qs)

                    prompt = conv.get_prompt()
                    input_ids = tokenizer_image_token(
                        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    ).unsqueeze(0).cuda()

                    with torch.inference_mode():
                        outputs = model(
                            input_ids=input_ids,
                            images=image_input,
                            image_sizes=[image.size],
                            agnostic_sparsity=args.agnostic_sparsity,
                            aware_sparsity=args.aware_sparsity,
                            use_cache=False,
                            output_attentions=True,
                            return_dict=True
                        )

                    # ---- IMPORTANT ----
                    # Replace these with exact indices from your repo once you confirm
                    seq_len = outputs.attentions[-1].shape[-1]

                    # Example placeholder split:
                    # text tokens = everything except the contiguous vision block
                    # vision tokens = known block in final merged sequence
                    vision_start = args.vision_start
                    vision_end = args.vision_end
                    vision_idx = torch.arange(vision_start, vision_end, device=outputs.attentions[-1].device)

                    text_left = torch.arange(0, vision_start, device=vision_idx.device)
                    text_right = torch.arange(vision_end, seq_len, device=vision_idx.device)
                    text_idx = torch.cat([text_left, text_right], dim=0)

                    stats = summarize_text_to_vision(outputs.attentions, text_idx, vision_idx)

                    record = {
                        "sample_id": idx,
                        "turn_id": turn_idx,
                        "prompt": prompt,
                        "seq_len": seq_len,
                        "vision_start": int(vision_start),
                        "vision_end": int(vision_end),
                        "stats": stats,
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()
          
                    del outputs
                    torch.cuda.empty_cache()

                elif speaker.upper() == "ASSISTANT":
                    if args.generate_responses:
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()
                        input_ids = tokenizer_image_token(
                            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                        ).unsqueeze(0).cuda()

                        generated_text = generate_assistant_reply(
                            args=args,
                            model=model,
                            tokenizer=tokenizer,
                            image_input=image_input,
                            image_size=image.size,
                            input_ids=input_ids,
                        )

                        conv.messages[-1][-1] = generated_text

                        if ans_f is not None:
                            ans_f.write(json.dumps({
                                "question_id": f"{idx}_{assistant_turn_counter}",
                                "sample_id": idx,
                                "turn_id": turn_idx,
                                "assistant_turn_id": assistant_turn_counter,
                                "prompt": conv.messages[-2][1] if len(conv.messages) >= 2 else "",
                                "text": generated_text,
                                "prediction": generated_text,
                                "reference": utterance,
                                "answer_id": shortuuid.uuid(),
                                "model_id": model_name,
                                "metadata": {}
                            }, ensure_ascii=False) + "\n")
                            ans_f.flush()

                        assistant_turn_counter += 1
                    else:
                        # For analysis-only experiments, use ground truth assistant reply
                        # so history stays aligned without generation
                        conv.append_message(conv.roles[1], utterance)

    if ans_f is not None:
        ans_f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument("--dataset-name", type=str, default="passing2961/MultiVerse")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache-dir", type=str, default=None)

    parser.add_argument("--output-file", type=str, required=True)

    parser.add_argument("--conv-mode", type=str, default="llava_v1")

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    # optional experiment controls
    parser.add_argument("--vision-start", type=int, default=0)
    parser.add_argument("--vision-end", type=int, default=0)
    parser.add_argument("--generate-responses", action="store_true")
    parser.add_argument("--answers-file", type=str, default="playground/data/eval/multiverse/answers/multiverse_experiment_answers.jsonl")
    parser.add_argument("--sample-size", type=str, default="all")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--agnostic-sparsity", type=float, default=0.0)
    parser.add_argument("--aware-sparsity", type=float, default=0.0)

    args = parser.parse_args()

    eval_model(args)
