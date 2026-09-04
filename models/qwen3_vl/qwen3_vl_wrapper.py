import numpy as np
import torch
import unicodedata

from dataclasses import dataclass
from tokenizers import processors
from PIL import Image
from typing import Optional, List, Union, Tuple

from transformers.modeling_outputs import ModelOutput
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.cache_utils import Cache
from transformers.utils.generic import check_model_inputs
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLPreTrainedModel, Qwen3VLModel, Qwen3VLConfig
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from qwen_vl_utils.vision_process import process_vision_info

FRAME_FACTOR=2

@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None

class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLForEmbeddingOutput]:
    
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


def sample_frames(frames, num_segments, max_segments):
    duration = len(frames)
    frame_id_array = np.linspace(0, duration-1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()
    last_frame_id = frame_id_list[-1]

    sampled_frames = []
    for frame_idx in frame_id_list:
        try:
            single_frame_path = frames[frame_idx]
        except:
            break
        sampled_frames.append(single_frame_path)
    # If total frame numbers is less than num_segments, append the last images to achieve
    while len(sampled_frames) < num_segments:
        sampled_frames.append(frames[last_frame_id])
    return sampled_frames[:max_segments]


class Qwen3VLForEmbeddingProcessor(Qwen3VLProcessor):
    
    def __init__(
        self, image_processor=None, tokenizer=None, 
        video_processor=None, chat_template=None, 
        eod_token='<|endoftext|>', max_length=16384,
        instruction_standalone: bool = True, 
        **kwargs
    ):
        super().__init__(
            image_processor=image_processor, tokenizer=tokenizer, 
            video_processor=video_processor, chat_template=chat_template, 
            **kwargs
        )

        if eod_token is not None:
            self.eod_id = self.tokenizer.convert_tokens_to_ids(eod_token)
        else:
            eod_token = self.tokenizer.eos_token
            self.eod_id = self.tokenizer.eos_token_id
        
        template_processor = processors.TemplateProcessing(
            single=f"$A {eod_token}", pair=f"$A $B {eod_token}", special_tokens=[(eod_token, self.eod_id)]
        )

        self.original_post_processor = self.tokenizer.backend_tokenizer.post_processor
        self.tokenizer.backend_tokenizer.post_processor = processors.Sequence(
            [self.tokenizer.backend_tokenizer.post_processor, template_processor]
        )

        self.instruction_standalone = instruction_standalone

        self.tokenizer.padding_side = kwargs.get('padding_side', 'right')
        self.max_length = max_length # NOTE: 暂时没用，使用的是方法中的 max_length
        self.min_pixels = self.image_processor.min_pixels or 32 * 32 * 4
        self.max_pixels = self.image_processor.max_pixels or 32 * 32 * 16384
        self.total_pixels = kwargs.get('total_pixels', 32 * 32 * 4500) # NOTE: 不用乘 2 因为后面乘了
        self.num_frames = self.video_processor.num_frames
        self.max_frames = self.video_processor.max_frames # NOTE: 默认值是 768
    
    def process(self, conversation, max_length=1024, truncation=False, padding=True):
        
        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        text = self.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        images, video_inputs, video_kwargs = process_vision_info(
            conversations, image_patch_size=16, 
            return_video_metadata=True, return_video_kwargs=True 
        )
        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None
        
        inputs = super().__call__(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            do_resize=False,
            return_tensors='pt',
            **video_kwargs # 包含 do_sample_frames=False
        )
        # breakpoint() # FIXME
        return inputs

    def __call__(
        self, texts: List[str], images: List[Union[str, Image.Image]] = None, videos: List[Union[str, Image.Image]]=None,
        is_query: bool = True, instruction: str | List[str] | None = None, **kwargs # include max_length
    ) -> BatchFeature:
        if isinstance(instruction, str):
            messages = [
                self.format_model_input(text, image, video, is_query, instruction) 
                for text, image, video in zip(texts, images, videos)
            ]
        else:
            messages = [
                self.format_model_input(text, image, video, is_query, _instruction) 
                for text, image, video, _instruction in zip(texts, images, videos, instruction)
            ]
        return self.process(messages, **kwargs)

    def format_model_input(
        self, text: str, image: Union[Image.Image, str], video: Union[str] = None,
        is_query: bool = True, instruction: str = None
    ):
        inputs = []

        # 处理instruction
        if instruction:
            instruction = instruction.strip()
            if instruction:
                if not unicodedata.category(instruction[-1]).startswith('P'):
                    instruction = instruction + '.'

        if self.instruction_standalone:
            system_input = {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": instruction if (is_query and instruction) else "Represent the user's input."
                    }
                ]
            }
            inputs.append(system_input)
        else:
            if is_query and instruction:
                if text:
                    text = instruction + ' ' + text
                else:
                    text = instruction


        # FIXME: 把下面 self. 相关的都检查一遍
        user_inputs = {
            "role": "user",
            "content": [
            ],
        }
        if not text and not image and not video:
            user_inputs['content'].append({'type': 'text', 'text': ""})
            inputs.append(user_inputs)
            return inputs
        if video:
            if isinstance(video, str):
                if video.startswith('http://') or video.startswith('https://'):
                    video_content = video
                else:
                    video_content = 'file://' + video
                video_process_kwargs = {'total_pixels': self.total_pixels}
            elif isinstance(video, list):
                video_content = video
                if self.num_frames is not None or self.max_frames is not None:
                    video_content = sample_frames(video_content, self.num_frames, self.max_frames)
                video_content = ['file://' + ele for ele in video_content]
                video_process_kwargs = {
                    'min_pixels': self.min_pixels,
                    'max_pixels': max(
                        min(self.max_pixels, self.total_pixels / len(video_content) * FRAME_FACTOR), 
                        int(self.min_pixels * 1.05)
                    )
                }
            if video_content:
                user_inputs['content'].append({
                    'type': 'video', 'video': video_content, 
                    **video_process_kwargs
                })
        if image:
            image_content = None
            if isinstance(image, Image.Image):
                image_content = image
            elif isinstance(image, str):
                if image.startswith('http://') or image.startswith('https://'):
                    image_content = image
                else:
                    image_content = 'file://' + image
            else:
                logger.error(f"image must be a PIL Image or a str, but got {type(image)}")
            if image_content:
                user_inputs['content'].append({
                    'type': 'image', 'image': image_content,
                    'max_pixels': self.max_pixels, 'min_pixels': self.min_pixels,
                })
        if text:
            user_inputs['content'].append({'type': 'text', 'text': text})
        inputs.append(user_inputs)

        return inputs

if __name__ == '__main__':
    model_path = './checkpoints/Qwen3-VL-Embedding-4B'
    model = Qwen3VLForEmbedding.from_pretrained(
        model_path, device_map='cuda', torch_dtype=torch.bfloat16
    )
    processor = Qwen3VLForEmbeddingProcessor.from_pretrained(
        model_path, instruction_standalone=True,
        max_length=1800,
        min_pixels=32*32*4,
        max_pixels=32*32*1280,
        total_pixels=32*32*4500,
        num_frames=48
    )

    texts = [
        "a woman breaks an egg",
        "a woman breaks two eggs in a bowl",
        None, None
    ]
    images = [None, None, None, None]
    videos = [
        None, None,
        './data/videos/WTf5EgVY5uU_98_104.avi',
        './data/videos/rw9h_574HxE_13_18.avi',
    ]
    instruction = [
        'Find the video snippet that corresponds to the given summary.',
        'Find the video snippet that corresponds to the given summary.',
        None, None
    ]

    inputs = processor(
        texts=texts, images=images, videos=videos, 
        instruction=instruction,
    ).to('cuda')
    outputs = model(**inputs)
    
    with torch.inference_mode():
        outputs = model(**inputs)
    
    embeddings = torch.nn.functional.normalize(
        outputs.last_hidden_state[
            torch.arange(outputs.last_hidden_state.shape[0]), 
            outputs.attention_mask.sum(dim=1) - 1
        ], 
        dim=-1
    )
    print(embeddings @ embeddings.T)
