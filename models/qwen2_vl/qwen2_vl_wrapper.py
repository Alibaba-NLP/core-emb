from typing import Optional, List, Union, Dict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor, Qwen2VLProcessorKwargs
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ImagesKwargs, VideosKwargs, ProcessingKwargs, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from qwen_vl_utils import process_vision_info
import logging

# from ..my_mteb_emb_model import register_model_type

logger = logging.getLogger(__name__)


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

def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask
    left_padding = (mask[:, -1].sum() == mask.shape[0])
    if left_padding:
        return hidden_state[:, -1]
    else:
        sequence_lengths = mask.sum(dim=1) - 1
        batch_size = hidden_state.shape[0]
        return hidden_state[torch.arange(batch_size, device=hidden_state.device), sequence_lengths]


class Qwen2VLForEmbedding(Qwen2VLForConditionalGeneration):

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        pooling_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        embs = _pooling_last(outputs.last_hidden_state, attention_mask)
        return embs


class Qwen2VLProcessorForEmbedding(Qwen2VLProcessor):

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, add_eos_id=True, max_length=1800, padding_side="left",
        min_pixels=4 * 28 * 28, max_pixels=1280 * 28 * 28, instruction_template=None, **kwargs
    ):
        super().__init__(image_processor, tokenizer, chat_template, **kwargs)

        self.add_eos_id = add_eos_id
        self.max_length = max_length
        self.padding_side = padding_side
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels  # H,W,P (patch_size * merge_size) * (patch_size * merge_size) * num_patches
        self.size = {
            'longest_edge': max_pixels,
            'shortest_edge': min_pixels
        }
        self.num_frames = kwargs.get('num_frames', None)
        self.max_frames = kwargs.get('max_frames', None)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if name == 'add_eos_id':
            self.eos_id = self.tokenizer.convert_tokens_to_ids('<|endoftext|>') if self.add_eos_id else None
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.eos_id
        if name == 'padding_side':
            self.tokenizer.padding_side = self.padding_side
        if name == 'min_pixels':
            self.image_processor.min_pixels = self.min_pixels
        if name == 'max_pixels':
            self.image_processor.max_pixels = self.max_pixels
        if name == 'size':
            self.image_processor.size = self.size

    def _call_processor(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        max_length: Optional[int] = None,
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Modified to add `eod_token` at the end of the text

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
            - **second_per_grid_ts** -- List of video seconds per time grid. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["images_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

            fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        self.video_token,
                        "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        # text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        if self.eos_id:
            tokenize_kwargs = {
                'truncation': True, 'max_length': max_length or self.max_length
            }
            tokenize_kwargs.update(output_kwargs['text_kwargs'])
            tokenize_kwargs.update({
                'padding': False, 'return_token_type_ids': False, 'return_tensors': None
            })
            text_inputs = self.tokenizer(text, **tokenize_kwargs)
            for seq, att in zip(text_inputs["input_ids"], text_inputs["attention_mask"]):
                seq.append(self.eos_id)
                att.append(1)
            pad_kwargs = {
                'padding': True, 'max_length': max_length or self.max_length
            }
            pad_kwargs.update({k: v for k, v in output_kwargs['text_kwargs'].items() if k != 'truncation'})
            text_inputs = self.tokenizer.pad(text_inputs, **pad_kwargs)
        else:
            tokenize_kwargs = {
                'padding': True, 'truncation': True, 'max_length': max_length or self.max_length,
            }
            tokenize_kwargs.update(output_kwargs['text_kwargs'])
            text_inputs = self.tokenizer(text, **tokenize_kwargs)
        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

    def format_model_input(
        self, text: str, image: Union[Image.Image, str], video: Union[str] = None,
        is_query: bool = True, instruction: str = None
    ):
        inputs = []
        if is_query and instruction:
            inputs.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": instruction
                    }
                ]
            })
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
            elif isinstance(video, list):
                video_content = video
                if self.num_frames is not None or self.max_frames is not None:
                    video_content = sample_frames(video_content, self.num_frames, self.max_frames)
                video_content = ['file://' + ele for ele in video_content]
            if video_content:
                user_inputs['content'].append({
                    'type': 'video',
                    'video': video_content,
                    'max_pixels': self.max_pixels,
                    'min_pixels': self.min_pixels,
                    'max_frames': self.num_frames,
                    'min_frames': self.num_frames,
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
                    'type': 'image',
                    'image': image_content,
                    'max_pixels': self.max_pixels,
                    'min_pixels': self.min_pixels,
                })
        if text:
            user_inputs['content'].append({'type': 'text', 'text': text})
        inputs.append(user_inputs)
        return inputs

    def process(
        self, messages: List[Dict], padding: bool = True,
        truncation: bool = True, return_tensors: str = 'pt', max_length: Optional[int] = None
    ) -> BatchFeature:
        texts = [
            self.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._call_processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors=return_tensors,
            text_kwargs={
                'padding': padding,
                'truncation': truncation,
            },
            max_length=max_length,
        )
        return inputs
        # return {'text': inputs, 'image': None}

    def __call__(
        self, texts: List[str], images: List[Union[str, Image.Image]] = None, videos: List[Union[str, Image.Image]] = None,
        is_query: bool = True, instruction: str | List[str] | None = None, **kwargs  # include max_length
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



# def load(model_name_or_path: str, **kwargs):
#     processor_kwargs = {}
#     if 'max_length' in kwargs:
#         processor_kwargs['max_length'] = kwargs.pop('max_length')
#     if 'padding_side' in kwargs:
#         processor_kwargs['padding_side'] = kwargs.pop('padding_side')
#     if 'add_eos_id' in kwargs:
#         processor_kwargs['add_eos_id'] = kwargs.pop('add_eos_id')
#     if 'add_eos_token' in kwargs:
#         processor_kwargs['add_eos_id'] = bool(kwargs.pop('add_eos_token'))
#     if 'min_pixels' in kwargs:
#         processor_kwargs['min_pixels'] = kwargs.pop('min_pixels')
#     if 'max_pixels' in kwargs:
#         processor_kwargs['max_pixels'] = kwargs.pop('max_pixels')
#     if 'instruction_template' in kwargs:
#         processor_kwargs['instruction_template'] = kwargs.pop('instruction_template')
#     if 'num_frames' in kwargs:
#         processor_kwargs['num_frames'] = kwargs.pop('num_frames')
#     if 'max_frames' in kwargs:
#         processor_kwargs['max_frames'] = kwargs.pop('max_frames')
#     model = Qwen2VLForEmbedding.from_pretrained(model_name_or_path, **kwargs)
#     model.eval()
#     processor = Qwen2VLProcessorForEmbedding.from_pretrained(
#         model_name_or_path, **processor_kwargs, **kwargs)

#     return model, processor
