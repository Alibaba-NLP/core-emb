from rzen_embed_inference import RzenEmbed
import os
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
query_inst = "Find me an everyday image that matches the given caption: "
psg_inst = "Represent the given image."
rzen = RzenEmbed(os.path.join(os.environ.get("MODEL_ROOT", "./checkpoints"), "RzenEmbed")).to("cuda")

import json
with open(os.path.join(os.environ.get("DATA_ROOT", "./data"), "benchmark", "scpp", "data", "replace_att.json"), 'r') as f:
    data = json.load(f)
cola_iter = iter(data)

item = next(cola_iter)
@torch.no_grad()
def emb(texts: list[str], is_query=True):
    if is_query:
        emb = rzen.get_fused_embeddings(instruction=query_inst,texts=texts)
    else:
        emb = rzen.get_fused_embeddings(instruction=psg_inst,images=texts)
    return emb

image_path = os.path.join(os.environ.get("DATA_ROOT", "./data"), "datasets", "coco2017_val", "val2017")
texts = [item['caption'], item['caption2'], item['negative_caption']]
images = [os.path.join(image_path, item['filename'])]

text_emb = emb(texts, is_query=True)
img_emb = emb(images, is_query=False)
sim = img_emb @ text_emb.T
print(sim)
