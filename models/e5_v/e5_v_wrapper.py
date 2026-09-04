from typing import List, Optional, Union
import torch
import torch.nn.functional as F
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image

# E5-V prompt template (LLaMA3 style)
E5V_LLAMA3_TEMPLATE = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n '


class E5VForEmbedding:
    """Wrapper class for E5-V embedding model.

    E5-V uses LLaVA-Next architecture with LLaMA3 prompt template for
    multimodal embedding. It extracts the last token's hidden state from
    the final layer and applies L2 normalization.
    """

    def __init__(self, model_path: str = 'royokong/e5-v', device: str = 'cuda'):
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(device)
        self.device = device
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_path: str, device_map: str = 'cuda', torch_dtype=None):
        """Load model from pretrained checkpoint.

        Args:
            model_path: Path to the model (default: 'royokong/e5-v')
            device_map: Device to load the model on
            torch_dtype: Data type (ignored, always uses float16)

        Returns:
            E5VForEmbedding instance
        """
        instance = cls.__new__(cls)
        instance.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(device_map)
        instance.device = device_map
        instance.model.eval()
        return instance

    def encode_image(self, images: List[Image.Image], prompt: str = None) -> torch.Tensor:
        """Encode images to embeddings.

        Args:
            images: List of PIL Images
            prompt: Optional custom prompt (default: '<image>\nSummary above image in one word: ')

        Returns:
            L2-normalized image embeddings [batch_size, hidden_dim]
        """
        if prompt is None:
            prompt = '<image>\nSummary above image in one word: '

        processor = E5VProcessorForEmbedding.get_processor()
        img_prompt = E5V_LLAMA3_TEMPLATE.format(prompt)
        inputs = processor(images=images, text=[img_prompt] * len(images),  return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings.

        Args:
            texts: List of text strings

        Returns:
            L2-normalized text embeddings [batch_size, hidden_dim]
        """
        processor = E5VProcessorForEmbedding.get_processor()
        text_prompt = E5V_LLAMA3_TEMPLATE.format('<sent>\nSummary above sentence in one word: ')
        inputs = processor([text_prompt.replace('<sent>', text) for text in texts], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        self.device = device
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self


class E5VProcessorForEmbedding:
    """Processor wrapper for E5-V model with configurable image size."""

    _processor = None

    def __init__(
        self,
        model_path: str = 'royokong/e5-v',
        max_pixels: int = 1280 * 28 * 28,
        min_pixels: int = 4 * 28 * 28,
        max_length: int = 1800,
    ):
        self.processor = LlavaNextProcessor.from_pretrained(model_path, patch_size=14)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_length = max_length
        self._configure_processor()

    def _configure_processor(self):
        """Configure image processor with size parameters."""
        size = {
            'longest_edge': self.max_pixels,
            'shortest_edge': self.min_pixels
        }
        self.processor.image_processor.size = size
        self.processor.image_processor.max_pixels = self.max_pixels
        self.processor.image_processor.min_pixels = self.min_pixels

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = 'royokong/e5-v',
        max_pixels: int = 1280 * 28 * 28,
        min_pixels: int = 4 * 28 * 28,
        max_length: int = 1800,
        **kwargs
    ):
        """Load processor from pretrained checkpoint.

        Args:
            model_path: Path to the model
            max_pixels: Maximum number of pixels for image resizing (default: 1280*28*28)
            min_pixels: Minimum number of pixels for image resizing (default: 4*28*28)
            max_length: Maximum sequence length (default: 1800)
            **kwargs: Additional arguments (ignored)

        Returns:
            E5VProcessorForEmbedding instance
        """
        instance = cls.__new__(cls)
        instance.processor = LlavaNextProcessor.from_pretrained(model_path, patch_size=14)
        instance.max_pixels = kwargs.get('max_pixels', max_pixels)
        instance.min_pixels = kwargs.get('min_pixels', min_pixels)
        instance.max_length = kwargs.get('max_length', max_length)
        instance._configure_processor()
        return instance

    @classmethod
    def get_processor(cls):
        """Get or create a shared processor instance."""
        if cls._processor is None:
            cls._processor = LlavaNextProcessor.from_pretrained('royokong/e5-v')
            # Set patch_size if not set (required for LlavaNextProcessor)
            if hasattr(cls._processor.image_processor, 'patch_size'):
                if cls._processor.image_processor.patch_size is None:
                    cls._processor.image_processor.patch_size = 14
            else:
                # For older transformers versions, try setting via config
                try:
                    cls._processor.image_processor.patch_size = 14
                except AttributeError:
                    pass
        return cls._processor

    def __call__(self, *args, **kwargs):
        return self.processor(*args, **kwargs)