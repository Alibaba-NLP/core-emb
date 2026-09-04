import torch
from PIL import Image
from typing import List, Union, Dict, Tuple, Optional, Literal
from dataclasses import dataclass
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor, Qwen2_5_VLProcessorKwargs
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.modeling_outputs import ModelOutput
from qwen_vl_utils import process_vision_info
import numpy as np


@dataclass
class Qwen2_5_VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.FloatTensor] = None


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


class Qwen2_5_VLForEmbedding(Qwen2_5_VLForConditionalGeneration):

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLForEmbeddingOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return outputs[0], attention_mask

        return Qwen2_5_VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


class Qwen2_5_VLProcessorForEmbedding(Qwen2_5_VLProcessor):

    # add video_processor for transformer v4.57.0
    def __init__(
        self, image_processor=None, tokenizer=None,chat_template=None, add_eos_id=True, max_length=1800, padding_side="left",
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
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
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
            Qwen2_5_VLProcessorKwargs,
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
            # sty add: for transformer v4.57.0
            # tokenize_kwargs.update({'return_token_type_ids' : False})
            # tokenize_kwargs.pop('return_mm_token_type_ids')

            text_inputs = self.tokenizer(text, **tokenize_kwargs)
            for seq, att in zip(text_inputs["input_ids"], text_inputs["attention_mask"]):
                seq.append(self.eos_id)
                att.append(1)
            pad_kwargs = {
                'padding': True, 'max_length': max_length or self.max_length
            }
            pad_kwargs.update({k: v for k, v in output_kwargs['text_kwargs'].items() if k != 'truncation'})
            # sty add: for transformer v4.57.0
            # pad_kwargs.update({"return_token_type_ids" : True})
            # pad_kwargs.pop("return_mm_token_type_ids")

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


if __name__ == '__main__':
    from transformers import AutoModel, AutoProcessor

    model_path = './checkpoints/Qwen25-VL-3B-v1'
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, add_eos_id=True)

    texts = [
        'Generate a descriptive caption for a throw pillow design influenced by traditional Asian art.\n',
        'a traditional Asian art',
        'a throw pillow'
    ]
    images = [None, None, None]

    inputs = processor(texts, images, return_tensors='pt')
    outputs = model(**inputs)
