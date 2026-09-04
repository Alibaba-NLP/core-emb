
import os
import torch
import transformers

from packaging.version import parse as V
from transformers import AutoProcessor, SiglipModel, AutoModel, AutoTokenizer
from transformers import CLIPTokenizerFast, CLIPImageProcessor

if transformers is not None and V(transformers.__version__) >= V("4.57.0"):
    from .qwen3_vl.qwen3_vl_wrapper import Qwen3VLForEmbedding, Qwen3VLForEmbeddingProcessor
    from .reranker.qwen3_vl_reranker_official import Qwen3VLReranker
    from swift.infer_engine import InferRequest, TransformersEngine
elif transformers is not None and V(transformers.__version__) > V("4.47.0"):
    Qwen3VLForEmbedding = None
    Qwen3VLForEmbeddingProcessor = None
    from .rzen.rzen_embed_inference import RzenEmbed
    from .qwen2_vl.qwen2_vl_wrapper import Qwen2VLForEmbedding, Qwen2VLProcessorForEmbedding
    from .qwen2_5_vl.embedding_qwen2_5_vl import Qwen2_5_VLForEmbedding, Qwen2_5_VLProcessorForEmbedding
    from transformers import MllamaForConditionalGeneration # This is for mme5
else:
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

MODEL_ROOT = os.environ.get("MODEL_ROOT", "./checkpoints")

def load_rzen(model_name: str, model_path: str, device: str = "cuda"):
    if model_path is None:
        model_path = os.path.join(MODEL_ROOT, "RzenEmbed")
    rzen = RzenEmbed(model_path).to(device)
    return [rzen] 

def load_qwen3_vl(model_name: str, model_path: str, device: str = "cuda"):
    model = Qwen3VLForEmbedding.from_pretrained(
        model_path, 
        device_map=device, 
        torch_dtype=torch.bfloat16
    )
    model.eval()
    processor = Qwen3VLForEmbeddingProcessor.from_pretrained(
        model_path, 
        instruction_standalone=True,
        max_length=3024,
        min_pixels=32*32*4,
        max_pixels=32*32*1280,
        total_pixels=32*32*4500,
        num_frames=48
    )
    return [model, processor]

def load_qwen2_vl(model_name: str, model_path: str, device: str = "cuda"):
    model = Qwen2VLForEmbedding.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = Qwen2VLProcessorForEmbedding.from_pretrained(
        model_path,
        instruction_standalone=True,
        max_length=3024,
        min_pixels=32*32*4,
        max_pixels=32*32*1280,
        total_pixels=32*32*4500,
        num_frames=48
    )
    return [model, processor]

def load_qwen2_5_vl(model_name: str, model_path: str, device: str = "cuda"):
    model = Qwen2_5_VLForEmbedding.from_pretrained(
        model_path, 
        device_map=device, 
        torch_dtype=torch.bfloat16
    )
    model.eval()
    processor = Qwen2VLProcessorForEmbedding.from_pretrained(
        model_path, 
        instruction_standalone=True,
        max_length=3024,
        min_pixels=32*32*4,
        max_pixels=32*32*1280,
        total_pixels=32*32*4500,
        num_frames=48
    )
    return [model, processor]

def load_triplet_clip(model_name: str, model_path: str, device: str = "cuda"):
    """
    Load TripletCLIP model with separate vision and text encoders.

    Args:
        model_name: Name of the model (used for identification)
        model_path: Path to the model (e.g., $MODEL_ROOT/CC12M_TripletCLIP_ViTB12)
        device: Device to load the model on

    Returns:
        List containing [vision_encoder, text_encoder, tokenizer, image_processor]
    """
    if model_path is None:
        model_path = os.path.join(MODEL_ROOT, "CC12M_TripletCLIP_ViTB12")

    # Load vision encoder from vision-encoder subfolder
    vision_encoder = AutoModel.from_pretrained(
        model_path,
        subfolder="vision-encoder",
        trust_remote_code=True
    ).to(device)
    vision_encoder.eval()

    # Load text encoder from text-encoder subfolder
    text_encoder = AutoModel.from_pretrained(
        model_path,
        subfolder="text-encoder",
        trust_remote_code=True
    ).to(device)
    text_encoder.eval()

    # Use CLIP tokenizer and image processor (model is based on openai/clip-vit-base-patch32)
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return [vision_encoder, text_encoder, tokenizer, image_processor]


def load_siglip(model_name: str, model_path: str, device: str = "cuda"):
    model = SiglipModel.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return [model, processor]


def load_e5_v(model_name: str, model_path: str = 'royokong/e5-v', device: str = "cuda"):
    """
    Load E5-V embedding model.

    Args:
        model_name: Name of the model (used for identification)
        model_path: Path to the model (default: 'royokong/e5-v')
        device: Device to load the model on

    Returns:
        List containing [model, processor]
    """
    if model_path is None:
        model_path = 'royokong/e5-v'

    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map=device,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = LlavaNextProcessor.from_pretrained(
        model_path,
        max_pixels=28*28*1280,
        min_pixels=28*28*4,
        max_length=3024
    )
    processor.patch_size=14
    return [model, processor]


def load_mm_e5(model_name: str, model_path: str = 'intfloat/mmE5-mllama-11b-instruct', device: str = "cuda"):
    """
    Load mmE5-mllama embedding model.

    Args:
        model_name: Name of the model (used for identification)
        model_path: Path to the model (default: 'intfloat/mmE5-mllama-11b-instruct')
        device: Device to load the model on

    Returns:
        List containing [model, processor]
    """
    if model_path is None:
        model_path = 'intfloat/mmE5-mllama-11b-instruct'

    model = MllamaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        max_pixels=28*28*1280,
        min_pixels=28*28*4,
        max_length=3024
    )
    return [model, processor]

def load_jina_reranker(model_name: str, model_path: str, device: str = "cuda"):
    """
    Load Jina reranker model.

    Args:
        model_name: Name of the model (used for identification)
        model_path: Path to the model (e.g., 'jinaai/jina-reranker-m0')
        device: Device to load the model on

    Returns:
        List containing [model]
    """
    if model_path is None:
        model_path = "jinaai/jina-reranker-m0"

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.to(device)
    model.eval()
    return [model]


def load_model(model_name: str, model_path: str = None, model_type: str = "embed", device: str = "cuda:0", ):
    if model_type == "embed":
        if 'rzen' in model_name:
            return load_rzen(model_name, model_path, device)
        elif 'vlembed' in model_name or "qwen3vl" in model_name or 'umarvel-qwen3vl-4b' in model_name:
            return load_qwen3_vl(model_name, model_path, device)
        elif 'seed' in model_name:
            return load_seed()
        elif model_name.find('ops') != -1 or model_name.find('gme') != -1 or model_name.find('unime') != -1 or model_name.find('umarvel-qwen2vl-7b') != -1 or 'vlm2vec' in model_name:
            return load_qwen2_vl(model_name, model_path, device)
        elif model_name.find("qwen25vl") != -1:
            return load_qwen2_5_vl(model_name, model_path, device)
        elif 'siglip' in model_name:
            return load_siglip(model_name, model_path, device)
        elif 'tripletclip' in model_name or 'triplet_clip' in model_name or 'negclip' in model_name:
            return load_triplet_clip(model_name, model_path, device)
        elif 'e5-v' in model_name or 'e5v' in model_name:
            return load_e5_v(model_name, model_path, device)
        elif 'mme5' in model_name.lower() or 'mm-e5' in model_name.lower():
            return load_mm_e5(model_name, model_path, device)
        elif model_path is not None:
            # Try to load as Qwen3-VL by default when model_path is provided
            print(f"Loading {model_name} from {model_path} as Qwen3-VL embedding model")
            return load_qwen3_vl(model_name, model_path, device)
        else:
            print(f"{model_name} not supported") 
    elif model_type == "reranker":
        if "qwen3vl" in model_name.lower() or "qwen35" in model_name.lower() or 'ours' in model_name.lower():
            return [TransformersEngine(
                model_path,
                task_type='generative_reranker',
                torch_dtype=torch.float16,
                attn_impl='flash_attention_2',
                device_map=device, 
                max_batch_size=32,
                max_length=1500
            )]
        elif 'jina' in model_name.lower():
            return load_jina_reranker(model_name, model_path, device)
        else:
            print(f"{model_name} not supported")
            return None
    else:
        print(f"{model_type} not support") 
    

