
import torch
import torch.nn.functional as F
from typing import List
from PIL import Image
query_inst = "Find me an everyday image that matches the given caption: "
psg_inst = "Represent the given image." 

@torch.no_grad()
def i2t_emb(model_name: str, model_list: List, texts: List[str], is_query=True):
    if len(model_list) == 2:
        model, processor = model_list
    elif len(model_list) == 1:
        model = model_list[0]
    elif len(model_list) == 3:
        # For models with separate encoders (e.g., TripletCLIP - old format)
        pass  # Will be unpacked in the specific model handling below
    elif len(model_list) == 4:
        # For TripletCLIP with separate tokenizer and image_processor
        pass  # Will be unpacked in the specific model handling below
    else:
        raise ValueError("model list error")

    if "rzen" in model_name: 
        if is_query:
            emb = model.get_fused_embeddings(instruction=query_inst,texts=texts)
        else:
            emb = model.get_fused_embeddings(instruction=psg_inst,images=texts)
    elif "vlembed" in model_name or 'qwen3vl' in model_name or 'qwen25vl' in model_name or 'umarvel-qwen3vl-4b' in model_name:
        if is_query:
            inputs = processor(texts=texts,  images=[None]*len(texts), videos=[None]*len(texts), instruction=[query_inst]*len(texts), is_query=False, max_length=1024, truncation=False)
        else:
            inputs = processor(texts=[None]*len(texts),  images=[Image.open(text) for text in texts], videos=[None]*len(texts), instruction=[psg_inst]*len(texts), is_query=False, max_length=1024, truncation=False)
        inputs = inputs.to(model.device)
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[
            torch.arange(outputs.last_hidden_state.shape[0]),
            outputs.attention_mask.sum(dim=1) - 1
        ]
    elif model_name.find('ops') != -1 or model_name.find('gme') != -1 or model_name.find('unime') != -1 or model_name.find('umarvel-qwen2vl-7b') != -1 or 'vlm2vec' in model_name:
        if is_query: 
            inputs = processor(texts=texts,  images=[None]*len(texts), videos=[None]*len(texts), instruction=[query_inst]*len(texts), is_query=False, max_length=1024, truncation=False)
        else: 
            inputs = processor(texts=[None]*len(texts),  images=[Image.open(text) for text in texts], videos=[None]*len(texts), instruction=[psg_inst]*len(texts), is_query=False, max_length=1024, truncation=False)
        inputs = inputs.to(model.device)
        emb = model(**inputs)
    elif 'seed' in model_name:
        if is_query:
            requests = processor(texts=texts, images=[None]*len(texts), videos=[None]*len(texts), instruction=[query_inst]*len(texts))
        else:
            requests = processor(texts=[None]*len(texts),  images=texts, videos=[None]*len(texts), instruction=[psg_inst]*len(texts))
        try:
            emb = model(requests).last_hidden_state.squeeze(1) # returns [1,1,2048]
        except Exception as e:
            print(e)
    elif 'siglip' in model_name:
        # max_length = getattr(model.config, 'text_config', {}).get('max_position_embeddings', 64)
        max_length = 64
        if is_query:
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items() if k != 'pixel_values'}
            emb = model.get_text_features(**inputs)
        else:
            images = [Image.open(text).convert("RGB") for text in texts]
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items() if k != 'input_ids' and k != 'attention_mask'}
            emb = model.get_image_features(**inputs)
    elif 'tripletclip' in model_name or 'triplet_clip' in model_name or 'negclip' in model_name:
        # TripletCLIP has separate vision and text encoders
        vision_encoder, text_encoder, tokenizer, image_processor = model_list
        max_length = 77  # CLIP default max length
        if is_query:
            # Text encoding using text encoder
            inputs = tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(text_encoder.device) for k, v in inputs.items()}
            outputs = text_encoder(**inputs)
            # Use text_embeds (projected output)
            emb = outputs.text_embeds
        else:
            # Image encoding using vision encoder
            images = [Image.open(text).convert("RGB") for text in texts]
            inputs = image_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(vision_encoder.device) for k, v in inputs.items()}
            # Vision encoder returns image_embeds directly
            emb = vision_encoder(inputs)
    elif 'e5-v' in model_name or 'e5v' in model_name:
        llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
        
        if is_query:
            text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')
            inputs =  processor([text_prompt.replace('<sent>', text) for text in texts], return_tensors="pt", padding=True).to(model.device)
        else:
            img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')
            images = [Image.open(text).convert("RGB") for text in texts]
            inputs = processor(images, [img_prompt]*len(texts),  return_tensors="pt", padding=True).to(model.device)
        emb = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
    elif 'mme5' in model_name.lower() or 'mm-e5' in model_name.lower():
        def last_pooling(last_hidden_state, attention_mask, normalize=False):
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
            if normalize:
                reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
            return reps
        if is_query:
            # Format query texts for image retrieval
            query_texts = [f'{query_inst}: {t}.\n' for t in texts]
            inputs = processor(text=query_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        else:
            images = [[Image.open(text).convert("RGB")] for text in texts]
            inputs = processor(text=len(texts)*['<|image|><|begin_of_text|>{Represent the given image.}.\n'], images=images, return_tensors='pt').to(model.device)
        emb = last_pooling(model(**inputs,return_dict=True, output_hidden_states=True).hidden_states[-1], inputs['attention_mask'])
    else:
        # Fallback: treat as Qwen3-VL style model (model_list = [model, processor])
        if len(model_list) == 2:
            model, processor = model_list
            if is_query:
                inputs = processor(texts=texts,  images=[None]*len(texts), videos=[None]*len(texts), instruction=[query_inst]*len(texts), is_query=False, max_length=1024, truncation=False)
            else:
                inputs = processor(texts=[None]*len(texts),  images=[Image.open(text) for text in texts], videos=[None]*len(texts), instruction=[psg_inst]*len(texts), is_query=False, max_length=1024, truncation=False)
            inputs = inputs.to(model.device)
            outputs = model(**inputs)
            emb = outputs.last_hidden_state[
                torch.arange(outputs.last_hidden_state.shape[0]),
                outputs.attention_mask.sum(dim=1) - 1
            ]
        else:
            raise ValueError(f"model name error: {model_name}")
    return emb

@torch.no_grad()
def fused_emb(model_name: str, model_list: List, texts: List[str], images: List[str]):
    """
    Generate fused embeddings from both text and image inputs.

    Args:
        model_name: Name of the model
        model_list: List containing model and processor
        texts: List of text strings
        images: List of image paths (same length as texts)

    Returns:
        Fused embeddings combining text and image information
    """
    if len(model_list) == 2:
        model, processor = model_list
    elif len(model_list) == 1:
        model = model_list[0]
    else:
        raise ValueError("model list error")

    assert len(texts) == len(images), f"texts and images must have same length, got {len(texts)} and {len(images)}"

    fused_inst = "Represent the given image and text."

    if "vlembed" in model_name or 'vlemb' in model_name or 'qwen3vl' in model_name or 'qwen25vl' in model_name or 'umarvel-qwen3vl-4b' in model_name:
        inputs = processor(
            texts=texts,
            images=[Image.open(img) for img in images],
            videos=[None] * len(texts),
            instruction=[fused_inst] * len(texts),
            is_query=False,
            max_length=4096,
            truncation=False
        )
        inputs = inputs.to(model.device)
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[
            torch.arange(outputs.last_hidden_state.shape[0]),
            outputs.attention_mask.sum(dim=1) - 1
        ]
    elif model_name.find('ops') != -1 or model_name.find('gme') != -1 or model_name.find('unime') != -1 or model_name.find('umarvel-qwen2vl-7b') != -1:
        inputs = processor(
            texts=texts,
            images=[Image.open(img) for img in images],
            videos=[None] * len(texts),
            instruction=[fused_inst] * len(texts),
            is_query=False,
            max_length=4096,
            truncation=False
        )
        inputs = inputs.to(model.device)
        emb = model(**inputs)
    elif 'seed' in model_name:
        requests = processor(
            texts=texts,
            images=images,
            videos=[None] * len(texts),
            instruction=[fused_inst] * len(texts)
        )
        try:
            emb = model(requests).last_hidden_state.squeeze(1)
        except Exception as e:
            print(e)
    elif "rzen" in model_name:
        emb = model.get_fused_embeddings(instruction=fused_inst, texts=texts, images=images)
    elif 'e5-v' in model_name or 'e5v' in model_name:
        # E5-V doesn't have native fused embedding, use image embedding with text context
        images_loaded = [Image.open(img).convert("RGB") for img in images]
        emb = model.encode_image(images_loaded)
    elif 'mme5' in model_name.lower() or 'mm-e5' in model_name.lower():
        # mmE5: encode each image with its text prompt separately then batch
        images_loaded = [Image.open(img).convert("RGB") for img in images]
        embs = []
        for img, t in zip(images_loaded, texts):
            text_prompt = f'<|begin_of_text|>Represent the given image with the following description: {t}\n'
            emb = model.encode_image([img], text_prompt)
            embs.append(emb)
        emb = torch.cat(embs, dim=0)
    else:
        raise ValueError(f"fused_emb not supported for model: {model_name}")

    return emb 