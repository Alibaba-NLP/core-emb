from .embedding_qwen2_5_vl import Qwen2_5_VLForEmbedding, Qwen2_5_VLProcessorForEmbedding
from ..my_mteb_emb_model import register_model_type

@register_model_type('qwen2_5_vl', ['text', 'image'])
def load(
    model_name_or_path: str, **kwargs
):
    processor_kwargs = {}
    if 'max_length' in kwargs:
        processor_kwargs['max_length'] = kwargs.pop('max_length')
    if 'padding_side' in kwargs:
        processor_kwargs['padding_side'] = kwargs.pop('padding_side')
    if 'add_eos_id' in kwargs:
        processor_kwargs['add_eos_id'] = kwargs.pop('add_eos_id')
    if 'add_eos_token' in kwargs:
        processor_kwargs['add_eos_id'] = bool(kwargs.pop('add_eos_token'))
    if 'min_pixels' in kwargs:
        processor_kwargs['min_pixels'] = kwargs.pop('min_pixels')
    if 'max_pixels' in kwargs:
        processor_kwargs['max_pixels'] = kwargs.pop('max_pixels')
    if 'instruction_template' in kwargs:
        processor_kwargs['instruction_template'] = kwargs.pop('instruction_template')
    if 'num_frames' in kwargs:
        processor_kwargs['num_frames'] = kwargs.pop('num_frames')
    if 'max_frames' in kwargs:
        processor_kwargs['max_frames'] = kwargs.pop('max_frames')
    model = Qwen2_5_VLForEmbedding.from_pretrained(model_name_or_path, **kwargs)
    model.eval()
    processor = Qwen2_5_VLProcessorForEmbedding.from_pretrained(
        model_name_or_path, **processor_kwargs, **kwargs)

    return model, processor
