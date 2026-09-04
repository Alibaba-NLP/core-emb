
from typing import List
import json
import torch
from embedding import i2t_emb
from tqdm import tqdm
from torch.nn import functional as F
from .composite_dataset import CompDataset
import transformers
from packaging.version import parse as V
from config import COLA_BENCHMARK_PATH
if transformers is not None and V(transformers.__version__) >= V("4.57.0"):
    from swift.infer_engine import InferRequest, TransformersEngine, RequestConfig


class ColaDataset(CompDataset):
    """
        metrics is the accuracy. 
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load()

    def load(
        self, dataset_path: str = COLA_BENCHMARK_PATH
    ):
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        self.data = data
        print("Load COLA successfully.")
    
    @torch.no_grad()
    def eval(self, model_name: str, model_list: List, dataset: str, batch_size: int = 32):
        score = 0
        predictions = {}
        for idx, item in enumerate(tqdm(self.data)):
            img0_emb, text0_emb = self.img_emb(model_name=model_name, model_list=model_list, img_paths= [item[0]]), self.text_emb(model_name=model_name, model_list=model_list, texts= [item[1]])
            img1_emb, text1_emb = self.img_emb(model_name=model_name, model_list=model_list, img_paths = [item[2]]), self.text_emb(model_name=model_name, model_list=model_list, texts=[item[3]])

            sim0 = float(img0_emb @ text0_emb.T)
            sim1 = float(img1_emb @ text1_emb.T)
            sim2 = float(img0_emb @ text1_emb.T)
            sim3 = float(img1_emb @ text0_emb.T)
            if sim0 > sim2 and sim1 > sim3:
                score += 1
                predictions[f"query-{idx+1}"] = {"acc" : 1}
            else:
                predictions[f"query-{idx+1}"] = {"acc" : 0}

        result = {
            'model_name' : model_name,
            'dataset' : dataset,
            'score' : "{:.4f}".format(float(score / len(self.data)))
        }

        return result, predictions

    def eval_reranker(self, model_name: str, model_list: List, dataset: str):
        if "jina" in model_name.lower():
            return self.eval_jina_reranker(model_name=model_name, model_list=model_list, dataset=dataset)
        # if "qwen3" in model_name:
        #     return self.eval_qwen3_vl_reranker(model_name=model_name, model_list=model_list, dataset=dataset)
        # else:
        #     return self.eval_ours_reranker(model_name=model_name, model_list=model_list, dataset=dataset)
        return self.eval_ours_reranker(model_name=model_name, model_list=model_list, dataset=dataset)

    @torch.no_grad()
    def eval_jina_reranker(self, model_name: str, model_list: List, dataset: str):
        score = 0
        predictions = {}
        model = model_list[0]  # get the reranker
        for idx, item in enumerate(tqdm(self.data)):
            # item[0], item[2] are image paths
            # item[1], item[3] are text captions
            # Pair structure: [text, image]
            pair1 = [[item[1], item[0]], [item[3], item[0]]]
            pair2 = [[item[1], item[2]], [item[3], item[2]]]

            scores1 = model.compute_score(pair1, max_length=2048, doc_type="image")
            scores2 = model.compute_score(pair2, max_length=2048, doc_type="image")

            # scores1[0]: similarity between text1 and image1
            # scores1[1]: similarity between text2 and image1
            # scores2[0]: similarity between text1 and image2
            # scores2[1]: similarity between text2 and image2
            # Correct matching: scores1[0] > scores2[0] and scores2[1] > scores1[1]
            if scores1[0] > scores2[0] and scores2[1] > scores1[1]:
                score += 1
                predictions[f"query-{idx+1}"] = {"acc": 1}
            else:
                predictions[f"query-{idx+1}"] = {"acc": 0}

        result = {
            'model_name': model_name,
            'dataset': dataset,
            'score': "{:.4f}".format(float(score / len(self.data)))
        }

        return result, predictions


    def eval_ours_reranker(self, model_name: str, model_list: List, dataset: str):
        score = 0
        predictions = {} 
        model = model_list[0] # get the reranker 
        requests = []
        # instruction = "Given a caption, determine if the image matches the description."
        # instruction = "Determine if the following image is relevant to the query. Answer only with 'yes' or 'no'."
        instruction = "Determine if the following image is relevant to the query."
        for idx, item in enumerate(tqdm(self.data)):
            pair1 = [(item[1], None, None, None, item[0], None), (item[3], None, None, None, item[0], None)]
            pair2 = [(item[1], None, None, None, item[2], None), (item[3], None, None, None, item[2], None)]
            
            requests.append(InferRequest(
                messages=[
                    { 'role': 'system', 'content': instruction }, 
                    { 'role': 'user', 'content': item[1]}, 
                    { 'role': 'assistant', 'content': "<image>"}
                ],
                images=[item[0]])
            )

            requests.append(InferRequest(
                messages=[
                    { 'role': 'system', 'content': instruction }, 
                    { 'role': 'user', 'content': item[3]}, 
                    { 'role': 'assistant', 'content': "<image>"}
                ],
                images=[item[0]])
            )

            requests.append(InferRequest(
                messages=[
                    { 'role': 'system', 'content': instruction }, 
                    { 'role': 'user', 'content': item[1]}, 
                    { 'role': 'assistant', 'content': "<image>"}
                ],
                images=[item[2]])
            )

            requests.append(InferRequest(
                messages=[
                    { 'role': 'system', 'content': instruction}, 
                    { 'role': 'user', 'content': item[3]}, 
                    { 'role': 'assistant', 'content': "<image>"}
                ],
                images=[item[2]])
            )

        request_config = RequestConfig(temperature=0, max_tokens=1500)
        
        responses = model.infer(requests, request_config)
        assert len(responses) == len(self.data) * 4
        for idx in range(0, len(self.data)):
            tmp_responses = responses[idx*4:idx*4+4]
            scores = [res.choices[0].message.content[0] for res in tmp_responses]
            # assert type for all in scores is float
            assert all([type(score) == float for score in scores])
            if scores[0] > scores[1] and scores[3]  >  scores[2]:
                score += 1
                predictions[f"query-{idx+1}"] = {"acc" : 1}
            else:
                predictions[f"query-{idx+1}"] = {"acc" : 0}

        result = {
            'model_name' : model_name,
            'dataset' : dataset,
            'score' : "{:.4f}".format(float(score / len(self.data)))
        }

        return result, predictions