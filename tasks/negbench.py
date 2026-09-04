from typing import List
import os
import json
import torch
from tqdm import tqdm
from .composite_dataset import CompDataset 
import pandas as pd
import os
import transformers
from packaging.version import parse as V
from config import COCO_VAL2017_PATH, VOC2007_PATH, NEGBENCH_DATA_PATH
if transformers is not None and V(transformers.__version__) >= V("4.57.0"):
    from swift.infer_engine import InferRequest, TransformersEngine, RequestConfig


class NegbenchDataset(CompDataset):
    """
        metrics is the accuracy. 
    """
    
    coco_image_path = COCO_VAL2017_PATH
    voc_image_path = VOC2007_PATH
    coco_neg_path = os.path.join(NEGBENCH_DATA_PATH, "COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv")
    coco_mcq_path = os.path.join(NEGBENCH_DATA_PATH, "COCO_val_mcq_llama3.1_rephrased.csv")
    voc_mcq_path = os.path.join(NEGBENCH_DATA_PATH, "VOC2007_mcq_llama3.1_rephrased.csv")
    voc_mcq_data, coco_neg_data, coco_mcq_data = [], [], []
    prediction_dict = {}
    results = {}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load()

    def load(self):
        coco_mcq_df = pd.read_csv(self.coco_mcq_path)
        for i, row in coco_mcq_df.iterrows():
            self.coco_mcq_data.append({
                "img_path" : os.path.join(self.coco_image_path, row['image_path']),
                "captions" : [row['caption_0'], row['caption_1'], row['caption_2'], row['caption_3']] # caption0 is the correct one
            })
        print("Loaded COCO MCQ dataset...")
        voc_mcq_df = pd.read_csv(self.voc_mcq_path)
        for i, row in voc_mcq_df.iterrows():
            self.voc_mcq_data.append({
                "img_path" : os.path.join(self.voc_image_path, row['image_path']),
                "captions" : [row['caption_0'], row['caption_1'], row['caption_2'], row['caption_3']] # caption0 is the correct one
            })
        print("Loaded VOC MCQ dataset...")
        coco_neg_df = pd.read_csv(self.coco_neg_path)
        for i, row in coco_neg_df.iterrows():
            self.coco_neg_data.append({
                "img_id" : row['image_id'],
                "img_path" : os.path.join(self.coco_image_path, row['filepath']),
                "captions" : eval(row['captions'])
            })
        print("Loaded COCO negated retrieval dataset...")
        self.data_dict = {
            'voc_mcq' : self.voc_mcq_data,
            'coco_neg' : self.coco_neg_data,
            'coco_mcq' : self.coco_mcq_data
        }    

    # dataset can only be voc_mcq, coco_mcq
    def eval_mcq(self, model_name, model_list, dataset: str, batch_size: int = 32): 
        score = 0.0 
        prediction = {}
        assert dataset in ['voc_mcq', 'coco_mcq'], "dataset must be voc_mcq or coco_mcq"
        to_eval_data = self.data_dict[dataset]
        # to_eval_data = to_eval_data[:10]
        imgs, texts = [], []
        for item in to_eval_data:
            imgs.append(item['img_path'])
            texts.extend(item['captions'])
        
        img_embs = self.img_batch_emb(model_name=model_name, model_list=model_list, img_paths=imgs, batch_size=batch_size).detach()
        text_embs = self.text_batch_emb(model_name=model_name, model_list=model_list, texts=texts, batch_size=batch_size).detach()

        for idx, item in enumerate(tqdm(to_eval_data)):
            img_emb = img_embs[idx].unsqueeze(0) 
            txt_emb = text_embs[idx * 4 : idx * 4 + 4]
            sims = img_emb @ txt_emb.T
            if sims[0][0] > sims[0][1] and sims[0][0] > sims[0][2] and sims[0][0] > sims[0][3]:
                hit = 1
            else:
                hit = 0
            score += hit
            prediction[idx] = {'acc' : hit}
        
        self.results[dataset] = {'score' : float(score/ len(to_eval_data))}
        self.prediction_dict[dataset] = prediction
        print(f"Evaluation of {dataset} on {model_name} done.")

    def _eval_mcq_reranker(self, model_name, model_list, dataset: str, batch_size: int = 32): 
        score = 0.0 
        prediction = {}
        model = model_list[0]
        assert dataset in ['voc_mcq', 'coco_mcq'], "dataset must be voc_mcq or coco_mcq"
        to_eval_data = self.data_dict[dataset]
        # to_eval_data = to_eval_data[:10]
        pairs = []
        for item in to_eval_data:
            for text in item['captions']:
                pairs.append([text, None, None, None, item['img_path'], None])
        
        rank_scores = model.predict(pairs, modality='vl')

        for idx, item in enumerate(tqdm(to_eval_data)):
          
            local_rank = rank_scores[idx * 4 : idx * 4 + 4]
            s1, s2, s3, s4 = local_rank
            if s1 > s2 and s1> s3 and s1>s4:
                hit = 1
            else:
                hit = 0
            score += hit
            prediction[idx] = {'acc' : hit}
        
        self.results[dataset] = {'score' : float(score/ len(to_eval_data))}
        self.prediction_dict[dataset] = prediction
        print(f"Evaluation of {dataset} on {model_name} done.")

    @torch.no_grad()
    def _eval_mcq_jina_reranker(self, model_name: str, model_list: List, dataset: str, batch_size: int = 32):
        """
        Evaluates a multiple-choice question (MCQ) reranker using Jina's compute_score method.
        """
        score = 0.0
        prediction = {}
        model = model_list[0]
        assert dataset in ['voc_mcq', 'coco_mcq'], "dataset must be voc_mcq or coco_mcq"
        to_eval_data = self.data_dict[dataset]

        all_rank_scores = []
        for item in tqdm(to_eval_data, desc=f"Jina reranking on {dataset}"):
            img_path = item['img_path']
            # Construct text-image pairs for all 4 captions
            pairs = [[caption, img_path] for caption in item['captions']]
            # Get scores from Jina reranker
            rank_scores = model.compute_score(pairs, max_length=2048, doc_type="image")
            all_rank_scores.extend(rank_scores)

        for idx, item in enumerate(tqdm(to_eval_data, desc="Calculating Accuracy")):
            local_rank = all_rank_scores[idx * 4 : idx * 4 + 4]
            s1, s2, s3, s4 = local_rank
            # caption_0 is the correct one
            if s1 > s2 and s1 > s3 and s1 > s4:
                hit = 1
            else:
                hit = 0
            score += hit
            prediction[idx] = {'acc': hit}

        self.results[dataset] = {'score': float(score / len(to_eval_data))}
        self.prediction_dict[dataset] = prediction
        print(f"Evaluation of {dataset} on {model_name} done.")

    def eval_coco_neg(self, model_name, model_list, dataset: str, batch_size: int =32):
        """
        Evaluates COCO negated retrieval with multiple captions per image.
        Each caption is treated as a separate query for text-to-image retrieval.
        """
        score = 0.0
        prediction = {}
        to_eval_data = self.data_dict[dataset]

        imgs, texts, text_to_img_map = [], [], []
        for item in to_eval_data:
            imgs.append(item['img_path'])
            for caption in item['captions']:
                texts.append(caption)
                text_to_img_map.append(item['img_id'])  # Map each caption to its image id

        img_embs = self.img_batch_emb(model_name=model_name, model_list=model_list, img_paths=imgs, batch_size=batch_size).detach()
        text_embs = self.text_batch_emb(model_name=model_name, model_list=model_list, texts=texts, batch_size=batch_size).detach()

        sims = text_embs @ img_embs.T  # t2i retrieval: (num_texts, num_imgs)
        img_ids = [item['img_id'] for item in to_eval_data]

        for text_idx, gt_img_id in enumerate(text_to_img_map):
            # Get similarity scores for this caption and find top-5 indices
            sim_scores = sims[text_idx]  # shape: (num_imgs,)
            top_5_indices = torch.topk(sim_scores, k=5).indices

            # Get top-5 image ids
            top_5_img_ids = [img_ids[idx] for idx in top_5_indices]

            # Check if the correct image is in top-5
            if gt_img_id in top_5_img_ids:
                recall_at_5 = 1
            else:
                recall_at_5 = 0
            prediction[text_idx] = {'acc': recall_at_5}
            score += recall_at_5

        self.results[dataset] = {'score': float(score / len(texts))}
        self.prediction_dict[dataset] = prediction
        print(f"Evaluation of coco_retrieval on {model_name} done.")

    def eval(self, model_name: str, model_list: List, dataset: str, batch_size: int =32):
        self.results = {
            'model_name' : model_name,
            'dataset' : dataset,
        }
        self.eval_mcq(model_name, model_list, "voc_mcq")
        self.eval_mcq(model_name, model_list, "coco_mcq")
        self.eval_coco_neg(model_name, model_list, "coco_neg")
        
        avg_score = 0.0
        for k, v in self.results.items():
            if k != 'model_name' and k != 'dataset':
                avg_score += v['score']
        
        avg_score = avg_score / 3
        self.results['avg_score'] = avg_score
        return self.results, self.prediction_dict
    
    def eval_reranker(self, model_name: str, model_list: List, dataset: str, batch_size: int =32):
        self.results = {
            'model_name' : model_name,
            'dataset' : dataset,
        }

        if "jina" in model_name.lower():
            self._eval_mcq_jina_reranker(model_name, model_list, "voc_mcq")
            self._eval_mcq_jina_reranker(model_name, model_list, "coco_mcq")
        else:
            # if "qwen3" in model_name:
            #     self._eval_mcq_reranker(model_name, model_list, "voc_mcq")
            #     self._eval_mcq_reranker(model_name, model_list, "coco_mcq")
            # else:
            #     self._eval_ours_mcq_reranker(model_name, model_list, "voc_mcq")
            #     self._eval_ours_mcq_reranker(model_name, model_list, "coco_mcq")
            # self.eval_coco_neg(model_name, model_list, "coco_neg")
            self._eval_ours_mcq_reranker(model_name, model_list, "voc_mcq")
            self._eval_ours_mcq_reranker(model_name, model_list, "coco_mcq")
        avg_score = 0.0
        for k, v in self.results.items():
            if k != 'model_name' and k != 'dataset':
                avg_score += v['score']
        
        avg_score = avg_score / 2
        self.results['avg_score'] = avg_score
        return self.results, self.prediction_dict
    

    def _eval_ours_mcq_reranker(self, model_name: str, model_list: List, dataset: str, batch_size: int = 1024):
        """
        Evaluates a multiple-choice question (MCQ) reranker using the .infer() method.
        It processes the data in batches to be memory-efficient.
        """
        model = model_list[0]
        assert dataset in ['voc_mcq', 'coco_mcq'], f"Unsupported dataset: {dataset}"
        to_eval_data = self.data_dict[dataset]
        
        # 1. 收集所有分数，分批次进行推理
        all_rank_scores = []
        instruction = "Determine if the following image is relevant to the query. "
        # instruction = "Given a caption, determine if the image matches the description."
        # 使用 tqdm 显示批处理进度
        for i in tqdm(range(0, len(to_eval_data), batch_size), desc=f"Inferencing on {dataset}"):
            batch_data = to_eval_data[i : i + batch_size]
            if not batch_data:
                continue
            
            batch_requests = []
            for item in batch_data:
                img_path = item['img_path']
                # 每个 item 有4个 captions
                for text in item['captions']:
                    batch_requests.append(InferRequest(
                        messages=[
                            {'role': 'system', 'content': instruction},
                            {'role': 'user', 'content': text},
                            {'role': 'assistant', 'content': "<image>"}
                        ],
                        images=[img_path]
                    ))
            
            # 对当前批次进行推理
            request_config = RequestConfig(temperature=0, max_tokens=1500)
            # input_ids = engine.template.encode(batch_requests)['input_ids']
            batch_responses = model.infer(batch_requests, request_config)
            # batch_responses = model.infer(batch_requests)

            # 解析分数并添加到总列表
            for res in batch_responses:
                try:
                    # 解析模型返回的分数
                    score = float(res.choices[0].message.content[0])
                    all_rank_scores.append(score)
                except (ValueError, TypeError) as e:
                    print(f"Error parsing score from model response: {e}")
                    print(f"Received content: {res.choices[0].message.content}")
                    all_rank_scores.append(0.0) # 添加一个默认值以保持长度一致
        
        assert len(all_rank_scores) == len(to_eval_data) * 4, "Mismatch between expected and received scores count."

        # 2. 评估逻辑 (与原函数相同)
        total_score = 0.0
        predictions = {}
        for idx, item in enumerate(tqdm(to_eval_data, desc="Calculating Accuracy")):
            # 从 all_rank_scores 中提取对应这个 item 的4个分数
            local_rank = all_rank_scores[idx * 4 : idx * 4 + 4]
            
            # 确保我们有4个分数可以解包
            if len(local_rank) == 4:
                s1, s2, s3, s4 = local_rank
                # 假设第一个 caption 是正确答案
                if s1 > s2 and s1 > s3 and s1 > s4:
                    hit = 1
                else:
                    hit = 0
            else:
                # 如果分数数量不匹配，则该样本判定为错误
                hit = 0
                
            total_score += hit
            # 使用 item 的唯一标识符作为 key 会更好，但这里遵循原逻辑使用 idx
            predictions[idx] = {'acc': hit}
        
        # 3. 保存结果 (与原函数相同)
        final_accuracy = total_score / len(to_eval_data) if to_eval_data else 0.0
        self.results[dataset] = {'score': final_accuracy}
        self.prediction_dict[dataset] = predictions
        
        print(f"Evaluation of {dataset} on {model_name} done. Accuracy: {final_accuracy:.4f}")
        return 
        # 为了与你其他的函数签名保持一致，也可以返回结果
        #return self.results, self.prediction_dict
