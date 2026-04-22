# multiverse_retrieval.py

import json
import math
import os
import shortuuid
import torch
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder_sparsity import load_pretrained_model
from llava.model.vision_retriever import VisionRetrievalConfig
from llava.utils import disable_torch_init


def build_first_user_prompt(user_text, model):
    from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    if model.config.mm_use_im_start_end:
        return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + user_text
    else:
        return DEFAULT_IMAGE_TOKEN + "\n" + user_text


def infer_patch_grid(model):
    # For CLIP ViT-L/14 at 336, this is 24x24 = 576 patch tokens.
    vt = model.get_vision_tower()
    nps = getattr(vt, "num_patches_per_side", None)
    if nps is None:
        return None
    return (nps, nps)


def build_retrieval_stats(vision_memory, retrieval_info):
    total_vision_tokens = int(vision_memory.full_tokens.size(0))
    selected_original_indices = retrieval_info.get("selected_original_indices", [])
    sink_original_indices = retrieval_info.get("sink_original_indices", [])
    selected_vision_tokens = len(selected_original_indices)
    sink_tokens_kept = len(sink_original_indices)
    kept_vision_token_pct = 0.0
    if total_vision_tokens > 0:
        kept_vision_token_pct = 100.0 * selected_vision_tokens / total_vision_tokens

    return {
        "total_vision_tokens": total_vision_tokens,
        "selected_vision_tokens": selected_vision_tokens,
        "sink_tokens_kept": sink_tokens_kept,
        "retrieved_non_sink_tokens": max(0, selected_vision_tokens - sink_tokens_kept),
        "kept_vision_token_pct": kept_vision_token_pct,
    }


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def run_multiturn_retrieval_eval(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        attn_implementation="eager",
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

    ds = load_dataset(args.dataset_name, split=args.split)
    if args.sample_size != "all":
        ds = ds.select(range(min(int(args.sample_size), len(ds))))

    samples = get_chunk(list(ds), args.num_chunks, args.chunk_idx)

    patch_grid = infer_patch_grid(model)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ans_f = None
    if args.answers_file:
        answers_path = os.path.expanduser(args.answers_file)
        answers_dir = os.path.dirname(answers_path)
        if answers_dir:
            os.makedirs(answers_dir, exist_ok=True)
        ans_f = open(answers_path, "w", encoding="utf-8")

    with open(args.output_file, "w", encoding="utf-8") as f:
        for sample in tqdm(samples):
            idx = sample["index"]
            image = sample["image"]
            if not isinstance(image, Image.Image):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")

            image_tensor = process_images([image], image_processor, model.config)[0]
            image_input = image_tensor.unsqueeze(0).half().cuda()

            # Build vision memory ONCE for the image
            vision_memory = model.build_single_image_vision_memory(
                images=image_input,
                vision_retrieval_config=retrieval_cfg,
                patch_grid=patch_grid,
                metadata={"sample_id": idx},
            )

            if args.use_projected_tokens_for_output:
                vision_memory = model.cache_projected_tokens_in_memory(vision_memory)

            conv = conv_templates[args.conv_mode].copy()
            speakers = sample["conversation"]["speaker"]
            utterances = sample["conversation"]["utterance"]
            checklist = sample["conversation"].get("checklist", [])
            assistant_turn_counter = 0
            last_query_info = None

            for turn_idx, (speaker, utterance) in enumerate(zip(speakers, utterances)):
                if speaker.upper() == "USER":
                    if len(conv.messages) == 0:
                        qs = build_first_user_prompt(utterance, model)
                    else:
                        qs = utterance

                    conv.append_message(conv.roles[0], qs)
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

                    with torch.inference_mode():
                        outputs = model(
                            input_ids=input_ids,
                            images=None,  # retrieval path uses vision_memory instead
                            use_cache=False,
                            output_attentions=False,
                            return_dict=True,
                            vision_memory=vision_memory,
                            vision_retrieval_config=retrieval_cfg,
                        )

                    # You can separately compute retrieval info if you want logging before model(...)
                    query_info = {
                        "sample_id": idx,
                        "turn_id": turn_idx,
                        "prompt": prompt,
                        "sink_original_indices": vision_memory.sink_original_indices.detach().cpu().tolist(),
                        "num_clusters": int(vision_memory.cluster_means.size(0)),
                        "num_kept_tokens": int(vision_memory.kept_original_indices.numel()),
                        **retrieval_stats,
                    }

                    f.write(json.dumps(query_info) + "\n")
                    f.flush()
                    last_query_info = query_info

                elif speaker.upper() == "ASSISTANT":
                    if args.generate_responses:
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()
                        input_ids = tokenizer_image_token(
                            prompt,
                            tokenizer,
                            IMAGE_TOKEN_INDEX,
                            return_tensors="pt",
                        ).unsqueeze(0).cuda()

                        with torch.inference_mode():
                            output_ids = model.generate(
                                inputs=input_ids,
                                images=None,
                                vision_memory=vision_memory,
                                vision_retrieval_config=retrieval_cfg,
                                do_sample=args.temperature > 0,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                num_beams=args.num_beams,
                                max_new_tokens=args.max_new_tokens,
                                use_cache=True,
                            )

                        output_ids = output_ids[:, input_ids.shape[1]:]
                        generated_text = tokenizer.batch_decode(
                            output_ids, skip_special_tokens=True
                        )[0].strip()

                        conv.messages[-1][-1] = generated_text

                        if ans_f is not None:
                            turn_checklist = checklist[turn_idx] if turn_idx < len(checklist) else {}
                            record = {
                                "question_id": f"{idx}_{assistant_turn_counter}",
                                "sample_id": idx,
                                "turn_id": turn_idx,
                                "assistant_turn_id": assistant_turn_counter,
                                "character": sample.get("character", ""),
                                "scenario": sample.get("scenario", ""),
                                "goal": sample.get("goal", ""),
                                "prompt": conv.messages[-2][1] if len(conv.messages) >= 2 else "",
                                "text": generated_text,
                                "prediction": generated_text,
                                "reference": utterance,
                                "checklist": turn_checklist,
                                "answer_id": shortuuid.uuid(),
                                "model_id": model_name,
                                "retrieval_stats": last_query_info,
                                "metadata": {},
                            }
                            ans_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            ans_f.flush()

                        assistant_turn_counter += 1
                    else:
                        conv.append_message(conv.roles[1], utterance)

    if ans_f is not None:
        ans_f.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)

    parser.add_argument("--dataset-name", type=str, default="passing2961/MultiVerse")
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, default=None)

    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--sample-size", type=str, default="20")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--generate-responses", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)

    # Retrieval params
    parser.add_argument("--prune-ratio", type=float, default=0.2)
    parser.add_argument("--num-sink-tokens", type=int, default=32)
    parser.add_argument("--num-clusters", type=int, default=16)
    parser.add_argument("--clustering-mode", type=str, default="kmeans", choices=["kmeans", "semantic"])
    parser.add_argument("--cluster-page-size", type=int, default=None)
    parser.add_argument("--topk-clusters", type=int, default=4)
    parser.add_argument("--retrieval-mode", type=str, default="cosine_mean")
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--use-projected-tokens-for-output", action="store_true")

    args = parser.parse_args()

    run_multiturn_retrieval_eval(args)
