from typing import List
import os
import json
import torch
from tqdm import tqdm
from .composite_dataset import CompDataset 
import transformers
from packaging.version import parse as V
from config import COCO_VAL2017_PATH, SCPP_DATA_PATH
if transformers is not None and V(transformers.__version__) >= V("4.57.0"):
    from swift.infer_engine import InferRequest, TransformersEngine, RequestConfig

class ScppDataset(CompDataset):
    """
        metrics is the accuracy. 
    """
    image_path = COCO_VAL2017_PATH

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load()

    def load(
        self, dataset_path: str = SCPP_DATA_PATH
    ):
        self.data_dict = {}
        for file in os.listdir(dataset_path):
            if file.endswith(".json"):
                sub_dataset_name = file.split(".")[0]
                with open(os.path.join(dataset_path, file), 'r') as f:
                    data = json.load(f)
                    self.data_dict[sub_dataset_name] = data
                print(f"Load SuperCrepe++: {sub_dataset_name} successfully.")
            else:
                continue
            
        print("Load SuperCrepe++ successfully.")

    def eval(self, model_name: str, model_list: List, dataset: str, batch_size: int =32, **kwargs):
        results = {
            'model_name' : model_name,
            'dataset' : dataset,
        }
        prediction_dict = {}
        for subset, subset_data in self.data_dict.items():
            
            imgs_path = []
            texts = []
            for item in subset_data:
                id = item['id']
                img_file_path = os.path.join(self.image_path, item['filename'])
                imgs_path.append(img_file_path) 
                cap1, cap2, neg = item['caption'], item['caption2'], item['negative_caption']
                texts.extend([cap1, cap2, neg])
             
            img_embs = self.img_batch_emb(model_name=model_name, model_list=model_list, img_paths=imgs_path, batch_size=batch_size).detach()
            text_embs = self.text_batch_emb(model_name=model_name, model_list=model_list, texts=texts, batch_size=batch_size).detach()

            score = 0.0
            predictions = {}
            for idx, item in enumerate(tqdm(subset_data)):
                id = item['id']
                img_emb = img_embs[idx].unsqueeze(0)
                text_emb = text_embs[idx * 3 : idx * 3 + 3]
                sims = img_emb @ text_emb.T
                sims = sims.to(torch.float32).cpu().numpy()
                if sims[0][0] > sims[0][2] and sims[0][1] > sims[0][2]:
                    hit = 1
                else:
                    hit = 0
                predictions[id] = {'acc' : hit}
                score = score + hit
            
            prediction_dict[subset] = predictions
            # results[subset]= {'score' : "{:.4f}".format(float(score / len(subset_data)))}
            results[subset]= {'score' : float(score / len(subset_data))}
            

        avg_score = 0.0
        for k, v in results.items():
            if k == 'dataset' or k == 'model_name':
                continue
            avg_score += v['score']
            results[k]['score'] = "{:.4f}".format(float(v['score']))
        
        avg_score = avg_score/5
        results['avg_score'] = "{:.4f}".format(float(avg_score))

        return results, prediction_dict
    

    def eval_reranker(self, model_name: str, model_list: List, dataset: str):
        if "jina" in model_name.lower():
            return self.eval_jina_reranker(model_name=model_name, model_list=model_list, dataset=dataset)
        # if "qwen3" in model_name:
        #     return self.eval_qwen3_vl_reranker(model_name=model_name, model_list=model_list, dataset=dataset)
        # else:
        #     return self.eval_ours_reranker(model_name=model_name, model_list=model_list, dataset=dataset)
        return self.eval_ours_reranker(model_name=model_name, model_list=model_list, dataset=dataset)


    def eval_qwen3_vl_reranker(self, model_name: str, model_list: List, dataset: str):
        results = {
            'model_name' : model_name,
            'dataset' : dataset,
        }
        model = model_list[0]
        prediction_dict = {}
        for subset, subset_data in self.data_dict.items():
            
            to_eval_pairs = []
            for item in subset_data:
                id = item['id']
                img_file_path = os.path.join(self.image_path, item['filename'])
                # imgs_path.append(img_file_path) 
                cap1, cap2, neg = item['caption'], item['caption2'], item['negative_caption']
                # texts.extend([cap1, cap2, neg])
                to_eval_pairs.append((cap1, None, None, None, img_file_path, None))
                to_eval_pairs.append((cap2, None, None, None, img_file_path, None))
                to_eval_pairs.append((neg, None, None, None, img_file_path, None))

            rank_scores = model.predict(to_eval_pairs, modality="vl") 
            score = 0.0
            predictions = {}
            for idx, item in enumerate(tqdm(subset_data)):
                id = item['id']
                local_rank = rank_scores[idx * 3 : idx * 3 + 3]
                if local_rank[0] > local_rank[2] and local_rank[1] > local_rank[2]:
                    hit = 1
                else:
                    hit = 0
                predictions[id] = {'acc' : hit}
                score = score + hit
            
            prediction_dict[subset] = predictions
            # results[subset]= {'score' : "{:.4f}".format(float(score / len(subset_data)))}
            results[subset]= {'score' : float(score / len(subset_data))}
            

        avg_score = 0.0
        for k, v in results.items():
            if k == 'dataset' or k == 'model_name':
                continue
            avg_score += v['score']
            results[k]['score'] = "{:.4f}".format(float(v['score']))
        
        avg_score = avg_score/5
        results['avg_score'] = "{:.4f}".format(float(avg_score))

        return results, prediction_dict

    @torch.no_grad()
    def eval_jina_reranker(self, model_name: str, model_list: List, dataset: str):
        results = {
            'model_name': model_name,
            'dataset': dataset,
        }
        model = model_list[0]
        prediction_dict = {}

        subsets_for_avg = []
        for subset, subset_data in self.data_dict.items():
            if not subset_data:
                continue

            all_rank_scores = []
            for item in tqdm(subset_data, desc=f"Jina reranking on {subset}"):
                img_file_path = os.path.join(self.image_path, item['filename'])
                texts = [item['caption'], item['caption2'], item['negative_caption']]

                # Construct text-image pairs
                pairs = [[text, img_file_path] for text in texts]
                rank_scores = model.compute_score(pairs, max_length=2048, doc_type="image")
                all_rank_scores.extend(rank_scores)

            score = 0.0
            predictions = {}
            for idx, item in enumerate(subset_data):
                id = item['id']
                local_rank = all_rank_scores[idx * 3 : idx * 3 + 3]
                # Both positive captions should score higher than negative
                if local_rank[0] > local_rank[2] and local_rank[1] > local_rank[2]:
                    hit = 1
                else:
                    hit = 0
                predictions[id] = {'acc': hit}
                score += hit

            prediction_dict[subset] = predictions

            if len(subset_data) > 0:
                subset_score = score / len(subset_data)
                results[subset] = {'score': subset_score}
                subsets_for_avg.append(subset_score)
            else:
                results[subset] = {'score': 0.0}

        # Calculate average score
        avg_score = 0.0
        if subsets_for_avg:
            avg_score = sum(subsets_for_avg) / len(subsets_for_avg)

        results['avg_score'] = "{:.4f}".format(avg_score)

        # Format all subset scores
        for key, value in results.items():
            if isinstance(value, dict) and 'score' in value:
                value['score'] = "{:.4f}".format(value['score'])

        return results, prediction_dict

    def eval_ours_reranker(self, model_name: str, model_list: List, dataset: str):
        results = {
            'model_name': model_name,
            'dataset': dataset,
        }
        model = model_list[0]
        prediction_dict = {}

        subsets_for_avg = []
        # instruction = "Given a caption, determine if the image matches the description."
        # instruction = "Determine if the following image is relevant to the query. Answer only with 'yes' or 'no'."
        instruction = "Determine if the following image is relevant to the query. "
        for subset, subset_data in self.data_dict.items():
            if not subset_data:
                continue

            # 1. 构建 InferRequest 列表，代替 to_eval_pairs
            requests = []
            for item in subset_data:
                img_file_path = os.path.join(self.image_path, item['filename'])
                texts = [item['caption'], item['caption2'], item['negative_caption']]
                
                for text_query in texts:
                    requests.append(InferRequest(
                        messages=[
                            # 使用和 eval_ours_reranker 相同的 prompt
                            {'role': 'system', 'content': instruction},
                            {'role': 'user', 'content': text_query},
                            {'role': 'assistant', 'content': "<image>"}
                        ],
                        images=[img_file_path]
                    ))
                   
            # 2. 对整个子集进行一次批处理调用
            request_config = RequestConfig(temperature=0, max_tokens=1500)
            responses = model.infer(requests, request_config)
            assert len(responses) == len(subset_data) * 3, "Number of responses should be 3 times the number of data items"

            score = 0.0
            predictions = {}
            for idx, item in enumerate(tqdm(subset_data, desc=f"Processing {subset}")):
                id = item['id']
                
                # 3. 从响应中解析分数
                tmp_responses = responses[idx * 3 : idx * 3 + 3]
                try:
                    # 假设模型返回的是可转换为 float 的字符串分数
                    # 使用 .strip() 增加代码健壮性
                    local_rank = [float(res.choices[0].message.content[0]) for res in tmp_responses]
                except (ValueError, TypeError) as e:
                    print(f"Error parsing score from model response: {e}")
                    print(f"Received contents: {[res.choices[0].message.content for res in tmp_responses]}")
                    # 如果解析失败，可以将该样本视为错误
                    local_rank = [0.0, 0.0, 1.0] # 确保比较失败

                # 4. 评估逻辑保持不变
                if local_rank[0] > local_rank[2] and local_rank[1] > local_rank[2]:
                    hit = 1
                else:
                    hit = 0
                
                predictions[id] = {'acc': hit}
                score += hit
            
            prediction_dict[subset] = predictions

            if len(subset_data) > 0:
                subset_score = score / len(subset_data)
                results[subset] = {'score': subset_score}
                subsets_for_avg.append(subset_score)
            else:
                results[subset] = {'score': 0.0}

        # 动态计算平均分
        avg_score = 0.0
        if subsets_for_avg:
            avg_score = sum(subsets_for_avg) / len(subsets_for_avg)
        
        results['avg_score'] = "{:.4f}".format(avg_score)

        # 格式化所有子集的分数
        for key, value in results.items():
            if isinstance(value, dict) and 'score' in value:
                value['score'] = "{:.4f}".format(value['score'])

        return results, prediction_dict

    def eval_ours_reranker_depre(self, model_name: str, model_list: List, dataset: str):
        results = {
            'model_name': model_name,
            'dataset': dataset,
        }
        model = model_list[0]
        prediction_dict = {}

        subsets_for_avg = []
        instruction = "Given a caption, determine if the image matches the description."
        for subset, subset_data in self.data_dict.items():
            if not subset_data:
                continue

            # 1. 构建 InferRequest 列表，代替 to_eval_pairs
            requests = []
            for item in subset_data:
                img_file_path = os.path.join(self.image_path, item['filename'])
                texts = [item['caption'], item['caption2'], item['negative_caption']]
                
                for text_query in texts:
                    requests.append(
                        {
                            "instruction":  instruction,
                            "query" : {"text": text_query },
                            "documents" : [
                                {"image": img_file_path},
                            ]         
                        }
                    )

            # 2. 对整个子集进行一次批处理调用
            # request_config = RequestConfig(temperature=0, max_tokens=1500)
            responses = []
            for request in tqdm(requests):
                responses.extend(model.process(request))
            # responses = model.process(requests)
            assert len(responses) == len(subset_data) * 3, "Number of responses should be 3 times the number of data items"

            score = 0.0
            predictions = {}
            for idx, item in enumerate(tqdm(subset_data, desc=f"Processing {subset}")):
                id = item['id']
                
                # 3. 从响应中解析分数
                tmp_responses = responses[idx * 3 : idx * 3 + 3]
                try:
                    # 假设模型返回的是可转换为 float 的字符串分数
                    # 使用 .strip() 增加代码健壮性
                    local_rank = [float(res.choices[0].message.content[0]) for res in tmp_responses]
                except (ValueError, TypeError) as e:
                    print(f"Error parsing score from model response: {e}")
                    print(f"Received contents: {[res.choices[0].message.content for res in tmp_responses]}")
                    # 如果解析失败，可以将该样本视为错误
                    local_rank = [0.0, 0.0, 1.0] # 确保比较失败

                # 4. 评估逻辑保持不变
                if local_rank[0] > local_rank[2] and local_rank[1] > local_rank[2]:
                    hit = 1
                else:
                    hit = 0
                
                predictions[id] = {'acc': hit}
                score += hit
            
            prediction_dict[subset] = predictions

            if len(subset_data) > 0:
                subset_score = score / len(subset_data)
                results[subset] = {'score': subset_score}
                subsets_for_avg.append(subset_score)
            else:
                results[subset] = {'score': 0.0}

        # 动态计算平均分
        avg_score = 0.0
        if subsets_for_avg:
            avg_score = sum(subsets_for_avg) / len(subsets_for_avg)
        
        results['avg_score'] = "{:.4f}".format(avg_score)

        # 格式化所有子集的分数
        for key, value in results.items():
            if isinstance(value, dict) and 'score' in value:
                value['score'] = "{:.4f}".format(value['score'])

        return results, prediction_dict