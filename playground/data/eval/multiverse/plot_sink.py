import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image
from argparse import Namespace
import json, os

from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def main():
    sample_id = 98
    dataset_name = "passing2961/MultiVerse"
    split = "train"
    layer = 'all'
    cache_dir = None

    aware_sparsity = .67
    agnostic_sparsity = .75

    predict_sink_sparsity = 1-(1-aware_sparsity) * (1-agnostic_sparsity)

    sample_size =20
    kept_file = f"multiverse_keeptokens_{sample_size}samplesize_{int(aware_sparsity*100)}aware_{int(agnostic_sparsity*100)}agnostic.jsonl"

    # Check kept file exists
    if kept_file is None or not os.path.isfile(kept_file):
        raise ValueError(f"Kept tokens file '{kept_file}' not found. Please provide the correct path.")

    TARGET_LAYER = -1
    SORT_BY_ATTENTION = True
    WEIGHTED = True

    ds = load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir
    )
    image = None

    for sample in ds:
        if str(sample["index"]) == str(sample_id):
            image = sample["image"]
            break
    else:
        raise ValueError(f"Sample {sample_id} not found.")
   
    tower = build_vision_tower()
  


    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")


    image.save(f"./sample_images/sample_image_{sample_id}.png")

    true_sink_tokens_local = []
    with open(kept_file) as f:
        for line in f:
            rec = json.loads(line)
           
            if rec["sample_id"] == str(sample_id) and rec.get("summary_type") == "common_across_queries":
                true_sink_tokens_local = rec["common_summary"][str(TARGET_LAYER)]["common_kept_token_indices"]
                break

    print(f"Number of sink tokens for sample {sample_id} at layer {TARGET_LAYER}: {len(true_sink_tokens_local)}")

    true_sink_sparsity = 1-(len(true_sink_tokens_local) / 576)

    info = (sample_id, agnostic_sparsity, aware_sparsity, predict_sink_sparsity, true_sink_sparsity, layer)
    
    
    attentions, hidden_states = encode_image(tower, image)

    agnostic_mask = agnostic_mask_tokens(attentions, agnostic_sparsity=agnostic_sparsity)
    
    agnostic_tokens = torch.where(agnostic_mask)[0].tolist()

    print(f"Number of agnostic tokens for sample {sample_id} at layer {TARGET_LAYER}: {len(agnostic_tokens)}")
    

    global_mask = torch.zeros(577)

   
    for (i, token) in enumerate(agnostic_tokens):
        if i in true_sink_tokens_local:
            global_mask[token] = 2
        else:
            global_mask[token] = 1

    true_sink_tokens = torch.where(global_mask == 2)[0].tolist()
    

    visualize_patch_mask(image, global_mask, info=info)

    predict_sink_tokens = predict_sink(attentions, predict_sink_sparsity, aware_sparsity)
    
    global_mask_predict = torch.zeros(577)
    for i in range(577):
        if i in predict_sink_tokens:
            global_mask_predict[i] = global_mask[i]+3
        else:
            global_mask_predict[i] = global_mask[i]
    visualize_patch_mask_predict(image, global_mask_predict, info=info)

    print(true_sink_tokens)
    print(predict_sink_tokens)
    hit_rate = calculate_hits_precision(true_sink_tokens, predict_sink_tokens)
    recall_rate = calculate_hits_recall(true_sink_tokens, predict_sink_tokens)
    f1_score = calculate_f1_score(hit_rate, recall_rate)

    print(f"Hits: {hit_rate}")
    print(f"Recall: {recall_rate}")
    print(f"F1 Score: {f1_score}")


    last_layer_attention = attentions[-1]
    cls_attention = last_layer_attention[:, :, 0, :].squeeze(0).mean(dim=0).squeeze(0)
    print(cls_attention.shape)
    plot_token_classification(cls_attention, true_sink_tokens, predict_sink_tokens, sort_by_attention=SORT_BY_ATTENTION, info=info, yscale='log')


def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def evaluate( sample_id, sink_tokens_local, tower, image, aware_sparsity, agnostic_sparsity, predict_sink_sparsity, print_visuals = False):


    true_sink_sparsity = 1-(len(sink_tokens_local) / 576)

    attentions, hidden_states = encode_image(tower, image)

    cls_attentions = [att[:, :, 0, :].squeeze(0) for att in attentions]
   
    

    agnostic_mask = agnostic_mask_tokens(cls_attentions, agnostic_sparsity=agnostic_sparsity)
    
    agnostic_tokens = torch.where(agnostic_mask)[0].tolist()

    global_mask = torch.zeros(577)

   
    for (i, token) in enumerate(agnostic_tokens):
        if i in sink_tokens_local:
            global_mask[token] = 2
        else:
            global_mask[token] = 1

    sink_tokens = torch.where(global_mask == 2)[0].tolist()

    predict_sink_tokens = predict_sink(cls_attentions, predict_sink_sparsity, aware_sparsity)


    
    
    hit_rate = calculate_hits_precision(sink_tokens, predict_sink_tokens)
    recall_rate = calculate_hits_recall(sink_tokens, predict_sink_tokens)
    f1_score = calculate_f1_score(hit_rate, recall_rate)

    info = {"sample_id": sample_id, "aware_sparsity": aware_sparsity, "agnostic_sparsity": agnostic_sparsity, "predict_sink_sparsity": predict_sink_sparsity, "true_sink_sparsity": true_sink_sparsity}

    if print_visuals:
        visualize_patch_mask(image, global_mask, info=info)

        global_mask_predict = torch.zeros(577)
        for i in range(577):
            if i in predict_sink_tokens:
                global_mask_predict[i] = global_mask[i]+3
            else:
                global_mask_predict[i] = global_mask[i]

        visualize_patch_mask_predict(image, global_mask_predict, info=info)

        plot_token_classification(cls_attentions[-1], sink_tokens, predict_sink_tokens, sort_by_attention=SORT_BY_ATTENTION, info=info, yscale='log')



    
    
  
    return hit_rate, recall_rate, f1_score









    

def calculate_hits_precision(true_sink_tokens, predict_sink_tokens):
    hits = 0
    for token in predict_sink_tokens:
        if token in true_sink_tokens:
            hits += 1
    return hits / len(predict_sink_tokens)

def calculate_hits_recall(true_sink_tokens, predict_sink_tokens):
    hits = 0
    for token in true_sink_tokens:
        if token in predict_sink_tokens:
            hits += 1
    return hits / len(true_sink_tokens)
  



def plot_token_classification(cls_attention, true_tokens, predicted_tokens, sort_by_attention=True, info=None, yscale="linear"):
    """
    cls_attention: tensor of shape [577] (CLS attention to all tokens)
    true_tokens: list of token indices (ground truth)
    predicted_tokens: tensor or list of token indices (predicted)

    Colors:
        green = correct (true & predicted)
        red   = missed (true only)
        blue  = false positive (predicted only)
        gray  = neither
    """

    # Convert to sets for fast lookup
    true_set = set(true_tokens)
    pred_set = set(predicted_tokens.tolist() if torch.is_tensor(predicted_tokens) else predicted_tokens)

    # Remove CLS token (index 0)
    scores = cls_attention[1:].detach().cpu().numpy()
    indices = np.arange(1, len(cls_attention))

    # Assign colors
    colors = []
    for idx in indices:
        if idx in true_set and idx in pred_set:
            colors.append("green")   # correct
        elif idx in true_set:
            colors.append("red")     # missed
        elif idx in pred_set:
            colors.append("blue")    # false positive
        else:
            colors.append("gray")    # neither

    # Optionally sort by attention
    if sort_by_attention:
        sorted_idx = np.argsort(scores)[::-1]  # descending
        scores = scores[sorted_idx]
        colors = [colors[i] for i in sorted_idx]

    # Plot
    plt.figure(figsize=(12, 4))
    plt.tight_layout()
    plt.bar(np.arange(len(scores)), scores, color=colors, width=1.0)

    plt.title("Token Classification vs CLS Attention")
    plt.xlabel("Tokens (sorted by attention)" if sort_by_attention else "Token Index")
    plt.ylabel("CLS Attention Score")
    if yscale == "log":
        plt.yscale("log")
  
    

    legend_handles = [
        mpatches.Patch(color='green', label='Correct (True ∩ Pred)'),
        mpatches.Patch(color='red', label='Missed True'),
        mpatches.Patch(color='blue', label='False Positive'),
        mpatches.Patch(color='gray', label='Neither')
    ]

    plt.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    if info is not None:
        plt.savefig(f"./hit_plots/hitplot_sample{info[0]}_aware{int(info[1]*100)}_agnostic{int(info[2]*100)}_true{int(info[3]*100)}_predicted{int(info[4]*100)}_yscale{yscale}.png")
    else:
        plt.savefig(f"./hit_plots/hitplot.png")
    plt.tight_layout()
    plt.show()

        

def build_vision_tower():
    args = Namespace(
        mm_vision_select_layer=-2,
        mm_vision_select_feature="patch"
    )

    tower = CLIPVisionTower(
        "openai/clip-vit-large-patch14-336",
        args
    )

    tower.load_model()
    tower.eval()

    return tower

def encode_image(tower, image):
    inputs = tower.image_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(tower.device)

    with torch.no_grad():
        outputs = tower.vision_tower(
            pixel_values,
            output_attentions=True,
            output_hidden_states=True
        )
    return outputs.attentions, outputs.hidden_states

def agnostic_mask_tokens(cls_attentions, agnostic_sparsity):

    
    cls_attention = cls_attentions[-1]


    
    cls_attention = cls_attention[-1,:]
    
    saliency_mask = cls_attention > cls_attention.quantile(agnostic_sparsity)
    


    return saliency_mask



def visualize_patch_mask(image, global_mask, info = None,patch_size=14, alpha=0.4):
    """
    image: PIL image (already RGB)
    global_mask: tensor of size 577 (including CLS token at index 0)
    patch_size: CLIP patch size (14 for ViT-L/14)
    alpha: transparency for overlays
    """

    img = np.array(image).copy()
    H, W, _ = img.shape

    # Remove CLS token → only patches
    patch_mask = global_mask[1:].cpu().numpy()  # size 576

    # Infer grid size (should be 24x24 for 336/14)
    num_patches = patch_mask.shape[0]
    grid_size = int(np.sqrt(num_patches))
    assert grid_size * grid_size == num_patches, "Mask is not square"

    patch_h = H // grid_size
    patch_w = W // grid_size

    overlay = img.copy()

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            val = patch_mask[idx]

            y0, y1 = i * patch_h, (i + 1) * patch_h
            x0, x1 = j * patch_w, (j + 1) * patch_w

            if val == 0:
                # Black out
                overlay[y0:y1, x0:x1] = 0

            elif val == 1:
                # Red overlay
                overlay[y0:y1, x0:x1] = (
                    (1 - alpha) * overlay[y0:y1, x0:x1] +
                    alpha * np.array([255, 0, 0])
                )

            elif val == 2:
                # Green overlay
                overlay[y0:y1, x0:x1] = (
                    (1 - alpha) * overlay[y0:y1, x0:x1] +
                    alpha * np.array([0, 255, 0])
                )

    overlay = overlay.astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    if info is not None:
        plt.title(f"Patch Mask Visualization\n"
            f"Sample {info[0]} - Aware {int(info[1]*100)}% - Agnostic {int(info[2]*100)}% - True Sink {int(info[3]*100)}%")
    else:
        plt.title("Patch Mask Visualization")
    plt.savefig(f"./patch_visualizations/patchmask_sample{info[0]}_aware{int(info[1]*100)}_agnostic{int(info[2]*100)}.png")
    plt.show()


def predict_sink(cls_attentions, sink_sparsity, aware_sparsity):
    

    token_num = cls_attentions[0].shape[1]

    counts = [0 for _ in range(token_num)]
    topk_aware = int(token_num * (1-aware_sparsity))
    topk_sink = int(token_num * (1-sink_sparsity))
    
    for layer in range(len(cls_attentions)):
        cls_attention = cls_attentions[layer]
        for head in range(cls_attention.shape[0]):

            cls_attention_head = cls_attention[head]
     
          
            hits = torch.topk(cls_attention_head, k = topk_aware).indices

            
            for hit in hits:
                counts[hit] += 1

            
    sink_mask = torch.topk(torch.tensor(counts), k = topk_sink).indices



    return sink_mask



def visualize_patch_mask_predict(image, global_mask, info = None,patch_size=14, alpha=0.4):
    """
    image: PIL image (already RGB)
    global_mask: tensor of size 577 (including CLS token at index 0)
    patch_size: CLIP patch size (14 for ViT-L/14)
    alpha: transparency for overlays
    """

    img = np.array(image).copy()
    H, W, _ = img.shape

    # Remove CLS token → only patches
    patch_mask = global_mask[1:].cpu().numpy()  # size 576

    # Infer grid size (should be 24x24 for 336/14)
    num_patches = patch_mask.shape[0]
    grid_size = int(np.sqrt(num_patches))
    assert grid_size * grid_size == num_patches, "Mask is not square"

    patch_h = H // grid_size
    patch_w = W // grid_size

    overlay = img.copy()

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            val = patch_mask[idx]

            y0, y1 = i * patch_h, (i + 1) * patch_h
            x0, x1 = j * patch_w, (j + 1) * patch_w

            if val == 0:
                # Black out
                overlay[y0:y1, x0:x1] = 0

            elif val == 1:
                # Red overlay
                overlay[y0:y1, x0:x1] = (
                    (1 - alpha) * overlay[y0:y1, x0:x1] +
                    alpha * np.array([255, 0, 0])
                )

            elif val == 2:
                # Green overlay
                overlay[y0:y1, x0:x1] = (
                    (1 - alpha) * overlay[y0:y1, x0:x1] +
                    alpha * np.array([0, 255, 0])
                )
            elif val == 3:
                # Blue overlay
                overlay[y0:y1, x0:x1] = (
                    (1 - alpha) * overlay[y0:y1, x0:x1] +
                    alpha * np.array([0, 0, 255])
                )
            elif val == 4:
                # Yellow overlay
                overlay[y0:y1, x0:x1] = (
                    (1 - alpha) * overlay[y0:y1, x0:x1] +
                    alpha * np.array([255, 255, 0])
                )
            elif val == 5:
                # Purple
                overlay[y0:y1, x0:x1] = (
                    (1 - alpha) * overlay[y0:y1, x0:x1] +
                    alpha * np.array([255, 0, 255])
                )
            else:
                print("error")



    black_patch = mpatches.Patch(color='black', label='Masked (0)')
    red_patch = mpatches.Patch(color='red', label='Agnostic (1)')
    green_patch = mpatches.Patch(color='green', label='Sink (2)')
    blue_patch = mpatches.Patch(color='blue', label='Predicted Sink, but true mask (3)')
    yellow_patch = mpatches.Patch(color='yellow', label='Predicted Sink, true agnostic (4)')
    purple_patch = mpatches.Patch(color='purple', label='Predicted Sink, true sink   (5)')


    


    overlay = overlay.astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.legend(
        handles=[black_patch, red_patch, green_patch, blue_patch, yellow_patch, purple_patch],
        loc='upper left',
        bbox_to_anchor=(1.02, 1),   # push legend outside right
        borderaxespad=0,
        frameon=False
    )
    plt.tight_layout()
    
    if info is not None:
        plt.title(f"Patch Mask Visualization\n"
            f"Sample {info[0]} - Aware {int(info[1]*100)}% - Agnostic {int(info[2]*100)}%\n"
            f"True Sink {int(info[4]*100)}% - Predicted Sink Sparsity {int(info[3]*100)}%")
    else:
        plt.title("Patch Mask Visualization")
    plt.savefig(f"./patch_visualizations/patchmask_prediction_sample{info[0]}_aware{int(info[1]*100)}_agnostic{int(info[2]*100)}_predicted{int(info[4]*100)}.png")
    plt.show()

def append_result(file_path, record):
    """
    Appends a single JSON record to a JSONL file.
    Creates file if it doesn't exist.
    """
    with open(file_path, "a") as f:
        f.write(json.dumps(record) + "\n")



if __name__ == "__main__": 
  
    sample_size = 20
    dataset_name = "passing2961/MultiVerse"
    split = "train"

    cache_dir = None

    aware_sparsity = .67
    agnostic_sparsity = .75

    predict_sink_sparsity = 1-(1-aware_sparsity) * (1-agnostic_sparsity)
 

 

    predict_sink_sparsity_variation = [1-predict_sink_sparsity*(.5**i) for i in range(5,1,-1)]




    TARGET_LAYER = -1
    SORT_BY_ATTENTION = True
    WEIGHTED = True

    kept_file = f"multiverse_keeptokens_{sample_size}samplesize_{int(aware_sparsity*100)}aware_{int(agnostic_sparsity*100)}agnostic.jsonl"

    if kept_file is None or not os.path.isfile(kept_file):
        raise ValueError(f"Kept tokens file '{kept_file}' not found. Please provide the correct path.")

    ds = load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir
    )

    ds = ds.select(range(sample_size))

    samples = {}
    for sample in ds:

        image = sample["image"]

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        samples[sample["index"]] = [image]


    


    tower = build_vision_tower()


    with open(kept_file) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("summary_type") == "common_across_queries":
                samples[rec["sample_id"]].append(rec["common_summary"][str(TARGET_LAYER)]["common_kept_token_indices"])

                
    output_file = "evaluation_results.jsonl"
    

    for sample in samples:
        for prediction in predict_sink_sparsity_variation:
            
            hit_rate, recall_rate, f1_score = evaluate(sample, samples[sample][1], tower, samples[sample][0], aware_sparsity, agnostic_sparsity, prediction)
            
            record = {
            "sample_id": sample,
            "prediction_sink_sparsity": float(prediction),
            "true_sink_sparsity": float(1 - len(samples[sample][1]) / 576),
            "aware_sparsity": float(aware_sparsity),
            "agnostic_sparsity": float(agnostic_sparsity),
            "hit_rate": float(hit_rate),
            "recall_rate": float(recall_rate),
            "f1_score": float(f1_score),
            }

            append_result(output_file, record)
            


    




        
