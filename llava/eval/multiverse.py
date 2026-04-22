#mulitverse.py
import argparse
import os
import json
import math
import types
import torch
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def build_first_user_prompt(sample, user_text, model):
    text = user_text
    if model.config.mm_use_im_start_end:
        return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
    else:
        return DEFAULT_IMAGE_TOKEN + "\n" + text


# collector to store per-query per-layer kept tokens
class SparseVILASelectionCollector:
    def __init__(self, vision_start, vision_end, sparsity):
        self.vision_start = vision_start
        self.vision_end = vision_end
        self.sparsity = sparsity
        self.current_query_results = {}
        self.current_query_id = None

    def start_query(self, query_id):
        self.current_query_id = query_id
        self.current_query_results = {}

    def save_layer_result(self, layer_idx, salience, keep_mask):
        # salience: [V], keep_mask: [V]
        kept_token_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1).tolist()

        self.current_query_results[layer_idx] = {
            "layer": int(layer_idx),
            "salience": salience.detach().float().cpu().tolist(),
            "keep_mask": keep_mask.detach().float().cpu().tolist(),
            "kept_token_indices": kept_token_indices,
        }

    def finish_query(self):
        return self.current_query_results


# resolve selected layers
def resolve_target_layers(num_hidden_layers, layer_spec=None):
    if layer_spec is None or layer_spec.strip() == "":
        return [0, num_hidden_layers // 2, num_hidden_layers - 1]

    layers = []
    for x in layer_spec.split(","):
        x = x.strip()
        if x == "":
            continue
        layers.append(int(x))
    layers = sorted(list(set(layers)))
    return layers


# patch only selected layers to compute SparseVILA-style token retention
def patch_llama_attention_layers(model, collector, target_layers):

    # locate the actual llama layers inside llava model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        decoder_layers = model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        decoder_layers = model.model.model.layers
    else:
        raise ValueError("Could not locate decoder layers. Please inspect model structure.")

    for layer_idx in target_layers:
        layer_module = decoder_layers[layer_idx]
        attn_module = layer_module.self_attn

        attn_module._original_forward = attn_module.forward
        attn_module._layer_index = layer_idx

        def patched_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            # standard LlamaAttention eager-style forward
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            # RoPE
            if "position_embeddings" in kwargs and kwargs["position_embeddings"] is not None:
                cos, sin = kwargs["position_embeddings"]
            else:
                kv_seq_len = key_states.shape[-2]
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )
            # no kv-cache for this analysis prefill
            if past_key_value is not None:
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx if hasattr(self, "layer_idx") else 0, {}
                )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # =========================
            # SparseVILA single-layer selection
            # attn_weights: [B, H, Q, K]
            # take visual keys only, aggregate over heads and query positions
            # =========================
            v_start = collector.vision_start
            v_end = collector.vision_end

            # clamp in case user-specified indices are out of range
            kv_len = attn_weights.shape[-1]
            v_start_clamped = max(0, min(v_start, kv_len))
            v_end_clamped = max(v_start_clamped, min(v_end, kv_len))

            if v_end_clamped > v_start_clamped:
                attn_vis = attn_weights[:, :, :, v_start_clamped:v_end_clamped]  # [B, H, Q, V]
                H = attn_vis.size(1)
                Q = attn_vis.size(2)

                # SparseVILA salience = sum over heads and query positions / (H * Q)
                salience = attn_vis.sum(dim=(1, 2)).squeeze(0) / (H * Q)  # [V]

                # threshold by quantile; sparsity=0.75 means keep roughly top 25%
                threshold = torch.quantile(salience.float(), collector.sparsity)
                keep_mask = salience > threshold

                collector.save_layer_result(self._layer_index, salience, keep_mask)

                del attn_vis, salience, keep_mask

            # continue normal attention forward
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value

        attn_module.forward = types.MethodType(patched_forward, attn_module)


def eval_model(args):
    disable_torch_init()

    aware = args.aware_sparsity
    agnostic = args.agnostic_sparsity

    vision_start = 35
    vision_end = int(vision_start + (576 * (1 - agnostic)))  # e.g. 35 + 576 * 0.25 = 179


    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        args.model_base,
        model_name,
        attn_implementation="eager",
    )

    # choose three layers by default: first / middle / last
    num_hidden_layers = model.config.num_hidden_layers
    target_layers = resolve_target_layers(num_hidden_layers, args.target_layers)

    # collector for sparsevila token selection
    collector = SparseVILASelectionCollector(
        vision_start=vision_start,
        vision_end=vision_end,
        sparsity=aware,
    )

    # patch only selected layers
    patch_llama_attention_layers(model, collector, target_layers)

    

    ds = load_dataset(
        args.dataset_name,
        split=args.split,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    sample_size = args.sample_size

    if sample_size != "all":
        sample_size = int(sample_size)
        ds = ds.select(range(sample_size))  # adjust or remove this line for full dataset

    samples = get_chunk(list(ds), args.num_chunks, args.chunk_idx)

    output_file = args.output_file + f"/multiverse_keeptokens_{sample_size}samplesize_{int(aware*100)}aware_{int(agnostic*100)}agnostic.jsonl"
    print(f"Output file: {output_file}")
    output_dir = os.path.dirname(output_file)
    print(f"Output directory: {output_file}")
    print(f"Outdir type: {type(output_dir)}")

    

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
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

            # collect all user-query layer results for this sample
            sample_query_records = []

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

                    # start collector for this query
                    collector.start_query(query_id=turn_idx)

                    with torch.inference_mode():
                        _ = model(
                            input_ids=input_ids,
                            images=image_input,
                            image_sizes=[image.size],
                            use_cache=False,
                            output_attentions=False,
                            return_dict=True,
                            agnostic_sparsity=agnostic,
                        )

                    layer_results = collector.finish_query()

                    query_record = {
                        "sample_id": idx,
                        "turn_id": turn_idx,
                        "prompt": prompt,
                        "target_layers": target_layers,
                        "vision_start": int(vision_start),
                        "vision_end": int(vision_end),
                        "aware_sparsity": float(args.aware_sparsity),
                        "agnostic_sparsity": float(args.agnostic_sparsity),
                        "layer_results": layer_results,
                    }
                    sample_query_records.append(query_record)

                    f.write(json.dumps(query_record) + "\n")
                    f.flush()

                    torch.cuda.empty_cache()

                elif speaker.upper() == "ASSISTANT":
                    # keep GT assistant reply in history
                    conv.append_message(conv.roles[1], utterance)

            # =========================
            # after all USER queries for this sample, compute common kept tokens across queries
            # for each target layer separately
            # =========================
            common_summary = {}
            for layer_idx in target_layers:
                keep_lists = []

                for qr in sample_query_records:
                    if str(layer_idx) in qr["layer_results"]:
                        keep_mask = torch.tensor(qr["layer_results"][str(layer_idx)]["keep_mask"], dtype=torch.bool)
                    elif layer_idx in qr["layer_results"]:
                        keep_mask = torch.tensor(qr["layer_results"][layer_idx]["keep_mask"], dtype=torch.bool)
                    else:
                        continue
                    keep_lists.append(keep_mask)

                if len(keep_lists) == 0:
                    continue

                keep_matrix = torch.stack(keep_lists, dim=0)  # [num_queries, V]
                common_kept = keep_matrix.all(dim=0)
                common_token_indices = torch.nonzero(common_kept, as_tuple=False).squeeze(-1).tolist()

                query_freq = keep_matrix.float().mean(dim=0)

                common_summary[layer_idx] = {
                    "common_kept_token_indices": common_token_indices,
                    "query_frequency": query_freq.tolist(),  # fraction of queries keeping each token
                }

            summary_record = {
                "sample_id": idx,
                "summary_type": "common_across_queries",
                "target_layers": target_layers,
                "vision_start": int(vision_start),
                "vision_end": int(vision_end),
                "aware_sparsity": float(args.aware_sparsity),
                "agnostic_sparsity": float(args.agnostic_sparsity),
                "common_summary": common_summary,
            }
            f.write(json.dumps(summary_record) + "\n")
            f.flush()
        print(f"Results saved to {output_file}")


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

    # selected layers, e.g. "0,16,31"; empty means first/middle/last
    parser.add_argument("--target-layers", type=str, default="")

    # SparseVILA sparsity quantile
    parser.add_argument("--aware-sparsity", type=float, default=0)
    parser.add_argument("--agnostic-sparsity", type=float, default=0)

    parser.add_argument("--sample-size", type=str, default=20)

    args = parser.parse_args()
    print('here')
    eval_model(args)