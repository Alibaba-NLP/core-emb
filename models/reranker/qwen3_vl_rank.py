import logging
from typing import Dict, Optional, List
import os

import json
import queue
import sys
from PIL import Image

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from typing import Union, List, Tuple, Any
import mteb
from mteb.model_meta import ModelMeta
import numpy as np
import torch
from torch import Tensor, nn
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F
from torch.utils.data._utils.worker import ManagerWatchdog
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, is_torch_npu_available, Qwen3VLForConditionalGeneration, Qwen3VLModel, AutoProcessor
from sentence_transformers import CrossEncoder, SentenceTransformer
import time
from torch.multiprocessing import Manager

torch.cuda.memory._set_allocator_settings('expandable_segments:False')
logger = logging.getLogger(__name__)


IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR # 4 tokens
MAX_PIXELS = 1280 * IMAGE_FACTOR * IMAGE_FACTOR # 1280 tokens
MAX_RATIO = 200

FRAME_FACTOR = 2
FPS = 1
MIN_FRAMES = 2
MAX_FRAMES = 64
MIN_TOTAL_PIXELS = 1 * FRAME_FACTOR * MIN_PIXELS # 1 帧
MAX_TOTAL_PIXELS = 4 * FRAME_FACTOR * MAX_PIXELS # 4 帧

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


class Qwen3VLRank(nn.Module):
    def __init__(self, model_path, token_true_id, token_false_id, peft_path=None, max_length=8192, attn_type='causal', format_type='chat', inference_type='yes_or_no'):
        super().__init__()
        # self.lm = Qwen3VLForConditionalGeneration.from_pretrained(model_path,  torch_dtype=torch.float16, attn_implementation="flash_attention_2", trust_remote_code=True)
        self.lm = Qwen3VLForConditionalGeneration.from_pretrained(model_path,  torch_dtype=torch.float16, attn_implementation="flash_attention_2", trust_remote_code=True).model

        # self.lm = AutoModelForCausalLM.from_pretrained(model_path,  torch_dtype=torch.float16, attn_implementation="flash_attention_2", trust_remote_code=True)
        # self.yes_or_no_linear = ""
        self.token_true_id = token_true_id
        self.token_false_id = token_false_id
        self.inference_type = inference_type
        self.lm.eval()
        self.score_linear = self.get_binary_linear(model_path)
        self.score_linear.eval()
    
    def get_binary_linear(self, model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name_or_path, trust_remote_code=True)

        token_yes = tokenizer.convert_tokens_to_ids("yes")
        token_no = tokenizer.convert_tokens_to_ids("no")

        lm_head_weights = model.lm_head.weight.data

        weight_yes = lm_head_weights[token_yes]
        weight_no = lm_head_weights[token_no]

        D = weight_yes.size()[0]
        linear_layer = torch.nn.Linear(D, 1, bias=False)
        with torch.no_grad():
            linear_layer.weight[0] = weight_yes - weight_no
        return linear_layer
    """
    @torch.no_grad()
    def process(self, inputs, **kwargs):
        batch_scores = self.lm(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)

        probs = torch.nn.functional.softmax(batch_scores, dim=1)
        scores = probs[:, 1].tolist()
        return scores
    """
    @torch.no_grad()
    def process(self, inputs):
        batch_scores = self.lm(**inputs).last_hidden_state[:,-1]
        scores = self.score_linear(batch_scores)
        scores = torch.sigmoid(scores).squeeze(-1).cpu().detach().tolist()
        return scores

        
class TokenizeWorker:
    def __init__(self, tokenizer_path, max_length=1024, qsize=4, format_type='chat', eod_token='<|im_end|>\n<|im_start|>assistant\n<think>\n', **kwargs):
    # def __init__(self, tokenizer_path, max_length=1024, qsize=4, format_type='chat', eod_token='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'):
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, padding_side='left')
        self.processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True, padding_side='left')
        self.eod_token = eod_token
        if self.eod_token is not None:
            self.eod_id_list = self.processor.tokenizer.encode(self.eod_token)
        else:
            self.eod_id_list = [self.processor.tokenizer.eos_token_id]
        self.max_length = max_length
        self.qsize = 4
        self.format_type = format_type





    def _tokenize_loop(self, input_queue, output_queue, device, shared_pool):
        while True:
            r = input_queue.get()
            if r is None:
                break
            n, batch = r
            inputs = self.tokenize(batch, device=device)

            # 存到共享池
            shared_pool[n] = inputs
            output_queue.put(n)  # 只传编号


                    
    def truncate_tokens_optimized(
        self,
        tokens: List[str], 
        max_length: int,
        special_tokens: List[str]
    ) -> List[str]:
        if len(tokens) <= max_length:
            return tokens

        special_tokens_set = set(special_tokens)

        # 1. 确定预算：计算我们能保留多少个非特殊token
        num_special = sum(1 for token in tokens if token in special_tokens_set)

        # 根据保证（特殊token总数 < max_length），这个值总是非负的
        num_non_special_to_keep = max_length - num_special

        # 2. 按预算构建最终列表
        final_tokens = []
        non_special_kept_count = 0
        for token in tokens:
            # 如果是特殊token，直接保留
            if token in special_tokens_set:
                final_tokens.append(token)
            # 如果是非特殊token，并且我们还有预算
            elif non_special_kept_count < num_non_special_to_keep:
                final_tokens.append(token)
                non_special_kept_count += 1
            # 如果是非特殊token但预算已用完，则丢弃（即什么都不做）

        return final_tokens

    def tokenize(self, pairs: list, **kwargs):   
        max_length = self.max_length
        """
        inputs = self.processor.apply_chat_template(
            pairs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        """
        text = self.processor.apply_chat_template(pairs, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(pairs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = self.processor(text=text, 
            images=images, 
            videos=videos,
            video_metadata=video_metadatas, 
            truncation=False,
            padding=False,
            max_length=max_length, 
            do_resize=False, 
            **video_kwargs)
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.truncate_tokens_optimized(inputs['input_ids'][i][:-5], max_length, self.processor.tokenizer.all_special_ids) + inputs['input_ids'][i][-5:]
        temp_inputs = self.processor.tokenizer.pad({'input_ids': inputs['input_ids']}, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in temp_inputs:
            inputs[key] = temp_inputs[key]
        return inputs
    
    """
    def _tokenize_loop(self, input_queue, output_queue, device):
        keep_queue = queue.Queue(self.qsize + 1)

        while True:
            r = input_queue.get()
            if r is None:
                break

            n, batch = r
            inputs = self.tokenize(batch)
            inputs.to(device)
            output_queue.put((n, inputs))
            if keep_queue.full():
                k = keep_queue.get()
                del k
            keep_queue.put(inputs)
            del r, n, batch, inputs

        while not keep_queue.empty():
            i = keep_queue.get()
            del i
        return
    """

def _encode_loop(model, input_queue, output_queue, device, shared_pool):
    model = model.to(device)
    watchdog = ManagerWatchdog()
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            while watchdog.is_alive():
                try:
                    n = input_queue.get()
                    if n is None:
                        break
                    inputs = shared_pool[n]  # 从共享池读
                    inputs.to(device)
                    results = model.process(inputs=inputs)
                    output_queue.put((n, results))
                except queue.Empty:
                    continue
"""
def _encode_loop(model, input_queue, output_queue, device, qsize=4):
    model = model.to(device)
    watchdog = ManagerWatchdog()
    keep_queue = queue.Queue(qsize + 1)

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            while watchdog.is_alive():
                r = input_queue.get()
                if r is None:
                    break
                n, inputs = r
                results = model.process(inputs=inputs)
                output_queue.put((n, results))
                if keep_queue.full():
                    i = keep_queue.get()
                    del i
                keep_queue.put(results)
                del r, n, inputs

    while not keep_queue.empty():
        i = keep_queue.get()
        del i
    del model, watchdog
    return
"""

class Qwen3VLRerankerInferenceModel(CrossEncoder):
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 8192,
        normalized: bool = False,
        qsize: int=4,
        instruction=None,
        format_type='chat',
        attn_type='causal',
        inference_type='yes_or_no',
        batch_size=16,
        **kwargs
    ) -> None:
        nn.Module.__init__(self)
        self.tokenizer = TokenizeWorker(model_name_or_path, max_length=max_length, format_type=format_type, **kwargs)
        self.model_name = model_name_or_path
        self.max_length=max_length
        self.qsize = qsize
        token_false_id = self.tokenizer.processor.tokenizer.get_vocab()["no"]
        token_true_id = self.tokenizer.processor.tokenizer.get_vocab()["yes"]
        self.instruction = instruction
        if self.instruction is None:
            self.instruction = "Given a web search query, retrieve relevant passages that answer the query."
        model = Qwen3VLRank(model_name_or_path, token_false_id=token_false_id, token_true_id=token_true_id, attn_type=attn_type, inference_type=inference_type)
        self.rank_model = model
        self.format_type = format_type
        self.batch_size = batch_size

        self.min_pixels = kwargs.get('min_pixels', MIN_PIXELS)
        self.max_pixels = kwargs.get('max_pixels', MAX_PIXELS)
        self.num_frames = kwargs.get('num_frames', None)
        self.max_frames = kwargs.get('max_frames', None)
        self.set_model_meta_data()
        task_prompts_path = "task_prompts_multilingual.json"
        # with open(task_prompts_path) as f:
        #     task_prompts = json.load(f)
        # self.task_prompts = task_prompts
        self.task_prompts = {}

    def set_model_meta_data(self):

        model_name = self.model_name
        if model_name.endswith('/'):
            model_name = model_name[:-1]
        model_name = '/'.join(model_name.split('/')[-2:])
        self.mteb_model_meta = ModelMeta(
            name=None, revision=None, release_date=None, languages=None,
            similarity_fn_name='cosine',
            license=None, open_weights=False, public_training_code=None, public_training_data=False, framework=['PyTorch'], use_instructions=None, training_datasets=None,
            embed_dim=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=self.max_length, modalities=['text', 'image']
        )
    def start(self):
        n_gpu = torch.cuda.device_count()

        self.world_size = n_gpu
        self.mp_ctx = torch.multiprocessing.get_context('spawn')
        self.shared_pool = Manager().dict()
        assert n_gpu > 0, 'woho, no no no!'
        logger.info(f"We have {n_gpu=}, good. Starting worker processes.")
        qsize = self.qsize
        self._text_queues = [self.mp_ctx.Queue(qsize) for _ in range(n_gpu)]
        self._input_queues = [self.mp_ctx.Queue(qsize) for _ in range(n_gpu)]
        self._output_queues = [self.mp_ctx.Queue(qsize) for _ in range(n_gpu)]
        self._devices = list()
        self._tokenize_wokers = list()
        self._encode_workers = list()
        self.shared_pool = Manager().dict()
        for i, (tq, iq, oq) in enumerate(zip(self._text_queues, self._input_queues, self._output_queues)):
            device = torch.device(f'cuda:{i}')
            self._devices.append(device)
            w_t = self.mp_ctx.Process(
                target=self.tokenizer._tokenize_loop, name=f'tok_w_{i}', args=(tq, iq, device, self.shared_pool)
            )
            w_t.start()
            self._tokenize_wokers.append(w_t)
            w_e = self.mp_ctx.Process(
                target=_encode_loop, name=f'enc_w_{i}', args=(self.rank_model, iq, oq, device, self.shared_pool)
            )
            w_e.start()
            self._encode_workers.append(w_e)
            logger.info(f"GPU {i} worker initiated.")

    """
    def __del__(self):
        self.stop()
    """
    def stop(self):
        for qs in (self._text_queues, self._input_queues):
            [q.put(None) for q in qs]
        for ws in (self._tokenize_wokers, self._encode_workers):
            [w.join() for w in ws]
            [w.close() for w in ws]
        for qs in (self._text_queues, self._input_queues, self._output_queues):
            [q.put(None) for q in qs]

    def format_mm_content(self, text, image, video, prefix='Query:'):
        content = []
        
        content.append({'type': 'text', 'text': prefix})
        if not text and not image and not video:
            content.append({'type': 'text', 'text': ""})
            return content
        if video:
            video_content = None
            if isinstance(video, list):
                video_content = video
                if self.num_frames is not None or self.max_frames is not None:
                    video_content = sample_frames(video_content, self.num_frames, self.max_frames)
                video_content = ['file://' + ele for ele in video_content]
            elif video.startswith('http') or video.startswith('oss'):
                video_content = video
            elif isinstance(video, str):
                video_content = 'file://' + video
            elif isinstance(video, list):
                video_content = ['file://' + ele for ele in video]
            if video_content:
                content.append({'type': 'video', 'video': video_content, 'total_pixels': MAX_TOTAL_PIXELS})
                # content.append({'type': 'video', 'video': video_content, 'total_pixels': MAX_TOTAL_PIXELS, 'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels})

        if image:
            image_content = None
            if isinstance(image, Image.Image):
                image_content = image

            elif image.startswith('http') or image.startswith('oss'):
                image_content = image
            elif isinstance(image, str):
                # image_content = 'file://' + image
                image_content = image
            else:
                image_content = image
            if image_content:
                content.append({'type': 'image', 'image': image_content,  "min_pixels": self.min_pixels, "max_pixels": self.max_pixels})

        if text:
            content.append({'type': 'text', 'text': text})
        return content 

    def format_mm_instruction(self, query_text, query_image, query_video, doc_text, doc_image, doc_video, instruction=None):
        inputs = []
        inputs.append({
            "role": "system",
            "content": [{
                "type": "text",
                "text": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            }
            ]
        })
        if isinstance(query_text, tuple):
            instruct, query_text = query_text
        else:
            instruct = instruction
        contents = []
        contents.append({
            "type": "text",
            "text": '<Instruct>: ' + instruct
        })
        query_content = self.format_mm_content(query_text, query_image, query_video, prefix='<Query>:')
        contents.extend(query_content)
        doc_content = self.format_mm_content(doc_text, doc_image, doc_video,  prefix='\n<Document>:')
        contents.extend(doc_content)
        inputs.append({
            "role": "user",
            "content": contents
        })
        return inputs 

    def format_mm_messages(self, pairs, instruction=None):
        instruction = instruction if instruction is not None else self.instruction
        messages = [self.format_mm_instruction(query, query_image, query_video, doc, doc_image, doc_video, instruction=instruction) for query, query_image, query_video, doc, doc_image, doc_video in pairs]
        return messages

    
    def get_instruction(
        self,
        task_name: str,
    ) -> str:
        """Get the instruction/prompt to be used for encoding sentences."""
        if task_name in self.task_prompts:
            if isinstance(self.task_prompts[task_name], dict):
                prompt = list(self.task_prompts[task_name].values())[0]
            else:
                prompt = self.task_prompts[task_name]
            return prompt
        task = mteb.get_task(task_name=task_name)
        task_metadata = task.metadata
        prompt = task_metadata.prompt
        if isinstance(prompt, dict):
            prompt = list(prompt.values())[0]
        return prompt

    def format_instruction(self, instruction, query, doc):
        if isinstance(query, tuple):
            instruction = query[0]
            query = query[1]
        text = [{"role": "system", "content": [{
                "type": "text",
                "text": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
                 }]},
                 {"role": "user", "content": [{"type": "text", "text": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}]}
        ]
        return text
    
    def _text_length(self, pair: list) -> int:
        # 这是一个估算长度的方法，可以基于内容的字符数
        try:
            # pair 的结构是复杂的 list of dicts
            return len(json.dumps(pair))
        except:
            return 0

    def predict(
        self,
        sentences: list[tuple[str, str]] | list[list[str]],
        batch_size: int = None,
        show_progress_bar: bool | None = True,
        num_workers: int = 1,
        activation_fct = None,
        apply_softmax: bool | None = False,
        convert_to_numpy: bool =  True,
        convert_to_tensor: bool = False,
        **kwargs
    ) -> list[torch.Tensor]:
        batch_size = batch_size or self.batch_size
        task_name = kwargs.get('task_name', None)
        if task_name is not None:
            instruction = self.get_instruction(task_name)
            # self.instruction = instruction
        else:
            instruction = kwargs.get('instruction', self.instruction)
        modality = kwargs.get('modality', 'text')
        if modality == 'text':
            pairs = [self.format_instruction(instruction, query, doc) for query, doc in sentences]
        else:
            pairs = [self.format_mm_instruction(query, query_image, query_video, doc, doc_image, doc_video, instruction=instruction) for query, query_image, query_video, doc, doc_image, doc_video in sentences]
        
        length_sorted_idx = np.argsort([-self._text_length(pair) for pair in pairs])
        pairs_sorted = [pairs[idx] for idx in length_sorted_idx]

        batch_size, num_texts = batch_size, len(pairs)
        num_batches = num_texts // batch_size + int(num_texts % batch_size > 0)
        def _receive(oq, timeout=0.00125):
            try:
                n, scores = oq.get(timeout=timeout)
                result_dict[n] = scores
                pbar.update(1)
                del scores
            except queue.Empty:
                pass
        show_progress_bar = show_progress_bar and (num_batches > 10)
        pbar = tqdm(total=num_batches, disable=not show_progress_bar, mininterval=1, miniters=10)
        result_dict = dict()
        for n, i in enumerate(range(0, num_texts, batch_size)):
            batch = pairs_sorted[i: i + batch_size]
            rank = n % self.world_size
            self._text_queues[rank].put((n, batch))
            if n >= self.world_size:
                _receive(self._output_queues[rank])
        while len(result_dict) < num_batches:
            for oq in self._output_queues:
                _receive(oq)

        pbar.close()
        results = []
        for n in range(len(result_dict)):
            results.extend(result_dict[n])
        results = [results[idx] for idx in np.argsort(length_sorted_idx)]

        return results


if __name__ == '__main__':
    model_path = "./checkpoints/Qwen3-VL-2B-Ranker"
    model = Qwen3VLRerankerInferenceModel(model_name_or_path=model_path, instruction="Retrieval document that can answer user's query", max_length=20480)

    model.start()
    # model.instruction = "Retrieval relevant document for the given query"
    # query = "Proving etiologic relationships to disease: the particular problem of human coronaviruses."
    # doc = "what is the origin of COVID-19"
    """
    query = "阿里巴巴总部在哪儿"
    docs = ["阿里巴巴总部在西溪园区","哈尔滨"]
    pairs = [(query, doc) for doc in docs]  * 100

    new_scores = model.predict(pairs)
    # print('text scores:', new_scores)
    """
    query = "老人在公园长椅上看书"
    query_image = None
    doc = None
    doc_image = "./data/benchmark/COLA/imgs/2390970.jpg"
    pairs = [(query, query_image, None, doc, doc_image, None)]*10
    # pairs = [(query, query_image, None, doc, doc_image, None),(query, query_image, None, doc, None, doc_video )]

    
    new_scores = model.predict(pairs, modality='vl')
    print('new_scores', new_scores)
    model.stop()
    exit()
