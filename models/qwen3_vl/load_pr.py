from .pr_wrapper import Qwen3VLForEmbedding, Qwen3VLForEmbeddingProcessor
from ..my_mteb_emb_model import register_model_type

@register_model_type('qwen3_vl_pr', ['text', 'image'])
def load_pr(
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
    if 'total_pixels' in kwargs:
        processor_kwargs['total_pixels'] = kwargs.pop('total_pixels')
    if 'instruction_template' in kwargs:
        processor_kwargs['instruction_template'] = kwargs.pop('instruction_template')
    if 'num_frames' in kwargs:
        processor_kwargs['num_frames'] = kwargs.pop('num_frames')
    if 'max_frames' in kwargs:
        processor_kwargs['max_frames'] = kwargs.pop('max_frames')
    if 'instruction_standalone' in kwargs:
        processor_kwargs['instruction_standalone'] = kwargs.pop('instruction_standalone')

    model = Qwen3VLForEmbedding.from_pretrained(model_name_or_path, **kwargs)
    model.eval()
    processor = Qwen3VLForEmbeddingProcessor.from_pretrained(
        model_name_or_path, **processor_kwargs, **kwargs)

    return model, processor