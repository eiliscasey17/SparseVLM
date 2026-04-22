#llava_arch.py
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

from llava.model.sparsity_experiments.sparseVila import generate_mask, generate_query_aware_mask, get_sink_tokens, compare_masks
from llava.model.sparsity_experiments.semanticClustering import register_key_hook, cluster_keys, mean_cluster_attention, generate_cluster_mask, prune_cluster

from dataclasses import dataclass

from llava.model.vision_memory import build_vision_memory, VisionMemory
from llava.model.vision_retriever import (
    VisionRetrievalConfig,
    build_text_query_representation,
    retrieve_visual_tokens_for_turn,
)

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def apply_query_aware_pruning(
        self,
        input_embeds,
        labels,
        image_token_ranges,
        aware_sparsity=0.0,
    ):
        if aware_sparsity <= 0 or len(image_token_ranges) == 0:
            return input_embeds, labels

        seq_len = input_embeds.shape[0]
        if seq_len == 0:
            return input_embeds, labels

        text_query_mask = torch.ones(seq_len, dtype=torch.bool, device=input_embeds.device)
        vision_positions = []
        for start, end in image_token_ranges:
            text_query_mask[start:end] = False
            vision_positions.append(torch.arange(start, end, device=input_embeds.device))

        if not vision_positions or not torch.any(text_query_mask):
            return input_embeds, labels

        vision_positions = torch.cat(vision_positions, dim=0)

        with torch.no_grad():
            prefill_outputs = self.get_model()(
                inputs_embeds=input_embeds.unsqueeze(0),
                attention_mask=torch.ones((1, seq_len), dtype=torch.bool, device=input_embeds.device),
                position_ids=torch.arange(seq_len, dtype=torch.long, device=input_embeds.device).unsqueeze(0),
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        attentions = prefill_outputs.attentions[-1]
        vision_scores = attentions[:, :, text_query_mask, :][:, :, :, vision_positions]
        vision_scores = vision_scores.sum(dim=(1, 2)).squeeze(0)
        keep_mask = generate_query_aware_mask(vision_scores, sparsity=aware_sparsity)

        kept_positions = vision_positions[keep_mask]

        rebuilt_embeds = []
        rebuilt_labels = []
        cursor = 0
        for start, end in image_token_ranges:
            if cursor < start:
                rebuilt_embeds.append(input_embeds[cursor:start])
                rebuilt_labels.append(labels[cursor:start])
            kept_image_positions = kept_positions[(kept_positions >= start) & (kept_positions < end)]
            if kept_image_positions.numel() > 0:
                rebuilt_embeds.append(input_embeds[kept_image_positions])
                rebuilt_labels.append(labels[kept_image_positions])
            cursor = end

        if cursor < seq_len:
            rebuilt_embeds.append(input_embeds[cursor:])
            rebuilt_labels.append(labels[cursor:])

        if not rebuilt_embeds:
            return input_embeds, labels

        return torch.cat(rebuilt_embeds, dim=0), torch.cat(rebuilt_labels, dim=0)
    
    def encode_images_raw(self, images):
        """
        Returns patch tokens before mm_projector, along with the last-layer attentions.
        Expects batch size 1 for now.
        """
        vision_tower = self.get_model().get_vision_tower()
        tower = vision_tower.vision_tower

        with torch.no_grad():
            output = tower(
                images,
                output_attentions=True,
                return_dict=True,
            )

        hidden = output.last_hidden_state   # [B, T, D]
        attentions = output.attentions[-1]  # [B, H, T, T]

        # drop CLS so patch tokens are indexed [0..N-1]
        patch_tokens = hidden[:, 1:, :]     # [B, N, D]

        projector_dtype = next(self.get_model().mm_projector.parameters()).dtype
        patch_tokens = patch_tokens.to(projector_dtype)

        return patch_tokens, attentions
    def build_single_image_vision_memory(
        self,
        images,
        vision_retrieval_config: VisionRetrievalConfig,
        patch_grid=None,
        metadata=None,
    ):
        """
        Builds one VisionMemory for one image.
        Expects images shape [1, C, H, W].
        """
        patch_tokens, attentions = self.encode_images_raw(images)  # [1,N,D], [1,H,T,T]
        assert patch_tokens.size(0) == 1, "Only batch size 1 supported in this scaffold."

        vm = build_vision_memory(
            patch_tokens=patch_tokens[0],
            attentions=attentions,
            prune_ratio=vision_retrieval_config.prune_ratio,
            num_sink_tokens=vision_retrieval_config.num_sink_tokens,
            num_clusters=vision_retrieval_config.num_clusters,
            clustering_mode=vision_retrieval_config.clustering_mode,
            cluster_page_size=vision_retrieval_config.cluster_page_size,
            patch_grid=patch_grid,
            metadata=metadata,
        )

        return vm
    
    def cache_projected_tokens_in_memory(self, vision_memory: VisionMemory):
        with torch.no_grad():
            proj = self.get_model().mm_projector(vision_memory.full_tokens)
        vision_memory.projected_full_tokens = proj

        from llava.model.vision_memory import rebuild_cluster_stats_from_projected_tokens
        vision_memory = rebuild_cluster_stats_from_projected_tokens(vision_memory)

        return vision_memory
    
    def retrieve_projected_tokens_for_query(
        self,
        vision_memory: VisionMemory,
        input_ids: torch.Tensor,
        image_token_index: int,
        vision_retrieval_config: VisionRetrievalConfig,
    ):
        """
        Build query vector from current input_ids, retrieve visual tokens, and return
        PROJECTED embeddings ready for insertion into the multimodal sequence.
        """
        query = build_text_query_representation(
            embed_tokens_module=self.get_model().embed_tokens,
            input_ids=input_ids,
            image_token_index=image_token_index,
        )  # [D_llm]

        # If vision token dim != text embed dim, add a simple linear map later.
        # For LLaVA, raw patch tokens and text embeds often live in different spaces.
        # Easiest baseline: use PROJECTED tokens for retrieval if cached.
        use_projected = vision_retrieval_config.use_projected_tokens_for_output

        if use_projected and vision_memory.projected_full_tokens is None:
            vision_memory = self.cache_projected_tokens_in_memory(vision_memory)

        retrieval_query = query
        token_source_is_projected = use_projected

        selected_tokens, info = retrieve_visual_tokens_for_turn(
            vision_memory=vision_memory,
            query=retrieval_query,
            topk_clusters=vision_retrieval_config.topk_clusters,
            retrieval_mode=vision_retrieval_config.retrieval_mode,
            token_budget=vision_retrieval_config.token_budget,
            use_projected_tokens=token_source_is_projected,
        )

        # if retrieved raw tokens, project now
        if not token_source_is_projected:
            selected_tokens = self.get_model().mm_projector(selected_tokens)

        return selected_tokens, info
    def prepare_inputs_labels_for_multimodal_with_retrieval(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        vision_memory: VisionMemory,
        vision_retrieval_config: VisionRetrievalConfig,
        image_token_index=IMAGE_TOKEN_INDEX,
    ):
        """
        Same return format as prepare_inputs_labels_for_multimodal(...)
        but inserts retrieved projected visual tokens instead of full/pruned image tokens.
        Expects batch size 1.
        """
        if input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove padding
        cur_input_ids = input_ids[0][attention_mask[0]]
        cur_labels = labels[0][attention_mask[0]]

        selected_image_features, retrieval_info = self.retrieve_projected_tokens_for_query(
            vision_memory=vision_memory,
            input_ids=cur_input_ids.unsqueeze(0),
            image_token_index=image_token_index,
            vision_retrieval_config=vision_retrieval_config,
        )  # [Nv, D]

        num_images = (cur_input_ids == image_token_index).sum().item()
        if num_images == 0:
            cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
            new_input_embeds = cur_input_embeds.unsqueeze(0)
            new_labels = cur_labels.unsqueeze(0)
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, retrieval_info

        image_token_indices = [-1] + torch.where(cur_input_ids == image_token_index)[0].tolist() + [cur_input_ids.shape[0]]

        cur_input_ids_noim = []
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])

        split_sizes = [x.shape[0] for x in cur_labels_noim]
        text_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        text_embeds_split = torch.split(text_embeds, split_sizes, dim=0)

        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(text_embeds_split[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_new_input_embeds.append(selected_image_features)
                cur_new_labels.append(
                    torch.full(
                        (selected_image_features.shape[0],),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
        cur_new_labels = torch.cat(cur_new_labels, dim=0)

        new_input_embeds = cur_new_input_embeds.unsqueeze(0)
        new_labels = cur_new_labels.unsqueeze(0)

        max_len = new_input_embeds.shape[1]
        attention_mask = torch.ones((1, max_len), dtype=torch.bool, device=new_input_embeds.device)
        position_ids = torch.arange(0, max_len, dtype=torch.long, device=new_input_embeds.device).unsqueeze(0)

        if _labels is None:
            new_labels = None
        if _attention_mask is not None:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, retrieval_info

    def encode_images(self, images, 
###################### Modify Here
                      agnostic_sparsity=.0):
########image_features = self.get_model().get_vision_tower()(images)
######## Modify Here

        vision_tower = self.get_model().get_vision_tower()
    
        tower = vision_tower.vision_tower

        layer_idx = vision_tower.select_layer 
        keys, handle = register_key_hook(tower,layer_idx)

        with torch.no_grad():
            output = tower(images,
                                output_attentions = True,
                                return_dict = True,)
        image_features = output.last_hidden_state

        keys_tensor = keys["k"]
        handle.remove()

        projector_dtype = next(self.get_model().mm_projector.parameters()).dtype
        image_features = image_features.to(projector_dtype)

     


       
        if agnostic_sparsity >0:
            attentions = output.attentions[-1]
            cls_attention  = attentions[:,:,0,]

            # Semantic Clustering
            '''clusters = cluster_keys(keys_tensor, num_clusters=4)
            cluster_attentions = mean_cluster_attention(clusters, cls_attention)
            sparsity_budget =torch.softmax(torch.tensor(cluster_attentions)* 1000, dim=0)
            mask = generate_cluster_mask(cls_attention, sparsity_budget= sparsity_budget, clusters=clusters, global_sparsity=agnostic_sparsity)'''
            
            # sparseVila
            mask = generate_mask(attentions[:,-1:, :1,], sparsity=agnostic_sparsity)

            # Semantic Batching
            #clusters = cluster_keys(keys_tensor, num_clusters=144)
            #mask = prune_cluster(cls_attention, clusters, agnostic_sparsity)
            
             # Semantic Batching
            #image_features = image_features[:, mask, :]
            
            # sparseVila or Semantic Clustering with cluster-level pruning
            image_features = image_features[:, mask.squeeze(0), :]

        #print("image_features shape:", image_features.shape)

        #print("mask shape:", mask.shape)
######## Modification End

        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, 
######## Modify Here
        agnostic_sparsity=0,
        aware_sparsity=0,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images, 
################################################ Modify Here
                                                agnostic_sparsity=agnostic_sparsity)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images, 
################################################ Modify Here
                                                agnostic_sparsity=agnostic_sparsity)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            image_token_ranges = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    image_start = sum(x.shape[0] for x in cur_new_input_embeds)
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    image_end = image_start + cur_image_features.shape[0]
                    image_token_ranges.append((image_start, image_end))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_input_embeds, cur_new_labels = self.apply_query_aware_pruning(
                cur_new_input_embeds,
                cur_new_labels,
                image_token_ranges,
                aware_sparsity=aware_sparsity,
            )

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
