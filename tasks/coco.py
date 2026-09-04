from typing import List
import os
import json
import torch
from tqdm import tqdm
from .composite_dataset import CompDataset
import pandas as pd
import pytrec_eval
from tqdm import tqdm
from config import COCO_VAL2017_PATH, NEGBENCH_DATA_PATH
class COCODataset(CompDataset):
    """
    COCO retrieval dataset for text-to-image and image-to-text retrieval.
    Metrics is Recall@1, Recall@5, Recall@10.
    """

    coco_image_path = COCO_VAL2017_PATH
    coco_retrieval_path = os.path.join(NEGBENCH_DATA_PATH, "COCO_val_retrieval.csv")
    coco_data = []
    prediction_dict = {}
    results = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load()

    def load(self):
        coco_df = pd.read_csv(self.coco_retrieval_path)
        for i, row in coco_df.iterrows():
            img_filename = f"{row['filepath']}"
            self.coco_data.append({
                "img_id": row['image_id'],
                "img_path": os.path.join(self.coco_image_path, img_filename),
                "captions": eval(row['captions'])
            })
        print(f"Loaded COCO retrieval dataset with {len(self.coco_data)} images...")

    def eval_coco_retrieval(self, model_name, model_list, dataset: str = "coco_retrieval", batch_size: int = 32, sim_batch_size: int = 1000):
        """
        Evaluates COCO retrieval with multiple captions per image.
        Supports both text-to-image and image-to-text retrieval.
        Uses memory-efficient batched similarity computation.
        """
        to_eval_data = self.coco_data

        imgs, texts, caption_counts, img_ids = [], [], [], []
        qrels_t2i = {}  # text-to-image: captions are queries, images are documents
        qrels_i2t = {}  # image-to-text: images are queries, captions are documents

        # Build qrels and collect data
        text_idx = 0
        for cid, item in enumerate(to_eval_data):
            imgs.append(item['img_path'])
            img_id = str(item['img_id'])
            img_ids.append(img_id)
            num_captions = len(item['captions'])
            caption_counts.append(num_captions)

            # For I2T: each image (query) maps to its captions (documents)
            qrels_i2t[img_id] = {str(text_idx + offset): 1 for offset in range(num_captions)}

            for caption in item['captions']:
                texts.append(caption)
                # For T2I: each caption (query) maps to its image (document)
                qrels_t2i[str(text_idx)] = {img_id: 1}
                text_idx += 1
        # breakpoint()
        img_embs = self.img_batch_emb(model_name=model_name, model_list=model_list, img_paths=imgs, batch_size=batch_size).detach()  # float16
        text_embs = self.text_batch_emb(model_name=model_name, model_list=model_list, texts=texts, batch_size=batch_size).detach() # float16

        num_imgs = len(imgs)
        num_texts = len(texts)
        # breakpoint()
        # Text-to-Image retrieval with batched computation
        t2i_predictions = {}
        for text_start in tqdm(range(0, num_texts, sim_batch_size), desc="Matrix computation"):
            text_end = min(text_start + sim_batch_size, num_texts)
            text_batch = text_embs[text_start:text_end]
            # Compute similarities for this batch
            sims_batch = text_batch @ img_embs.T  # (batch_size, num_imgs)

            for local_idx, text_idx in enumerate(range(text_start, text_end)):
                sim_scores = sims_batch[local_idx]
                # Only keep top-100 scores to reduce memory
                top_k = min(100, num_imgs)
                top_indices = torch.topk(sim_scores, top_k).indices.tolist()
                pred = {img_ids[idx]: float(sim_scores[idx]) for idx in top_indices}
                t2i_predictions[str(text_idx)] = pred

            del sims_batch

        # Evaluate T2I
        evaluator_t2i = pytrec_eval.RelevanceEvaluator(
            qrels_t2i,
            {"map_cut", "ndcg_cut",
             "recall_1", "recall_3", "recall_5", "recall_10",
             "recall_50", "recall_100"}
        )
        t2i_results = evaluator_t2i.evaluate(t2i_predictions)

        # Debug: check if keys match
        # print(f"Sample qrels_t2i keys (first 3): {list(qrels_t2i.keys())[:3]}")
        # print(f"Sample qrels_t2i values (first 3): {[qrels_t2i[k] for k in list(qrels_t2i.keys())[:3]]}")
        # print(f"Sample t2i_predictions keys (first 3): {list(t2i_predictions.keys())[:3]}")
        # print(f"Sample t2i_predictions values (first 3): {[t2i_predictions[k] for k in list(t2i_predictions.keys())[:3]]}")

        # Aggregate T2I results
        t2i_metrics = {}
        for metric in ["map_cut_10", "ndcg_cut_5", "ndcg_cut_10",
                       "recall_1", "recall_3", "recall_5", "recall_10"]:
            t2i_metrics[metric] = sum(r[metric] for r in t2i_results.values()) / len(t2i_results)

        # Image-to-Text retrieval with batched computation
        i2t_predictions = {}
        for img_start in range(0, num_imgs, sim_batch_size):
            img_end = min(img_start + sim_batch_size, num_imgs)
            img_batch = img_embs[img_start:img_end]

            # Compute similarities for this batch
            sims_batch = img_batch @ text_embs.T  # (batch_size, num_texts)

            for local_idx, img_idx in enumerate(range(img_start, img_end)):
                sim_scores = sims_batch[local_idx]
                # Only keep top-100 scores to reduce memory
                top_k = min(100, num_texts)
                top_indices = torch.topk(sim_scores, top_k).indices.tolist()
                pred = {str(qid): float(sim_scores[qid]) for qid in top_indices}
                i2t_predictions[img_ids[img_idx]] = pred

            del sims_batch

        # Evaluate I2T
        evaluator_i2t = pytrec_eval.RelevanceEvaluator(
            qrels_i2t,
            {"map_cut", "ndcg_cut",
             "recall_1", "recall_3", "recall_5", "recall_10",
             "recall_50", "recall_100"}
        )
        i2t_results = evaluator_i2t.evaluate(i2t_predictions)

        # Aggregate I2T results
        i2t_metrics = {}
        for metric in ["map_cut_10", "ndcg_cut_5", "ndcg_cut_10",
                       "recall_1", "recall_3", "recall_5", "recall_10"]:
            i2t_metrics[metric] = sum(r[metric] for r in i2t_results.values()) / len(i2t_results)

        # Store results
        self.results['t2i'] = t2i_metrics
        self.results['i2t'] = i2t_metrics

        print(f"Evaluation of COCO retrieval on {model_name} done.")
        print(f"T2I Recall@1/3/5/10: {t2i_metrics['recall_1']:.4f}/{t2i_metrics['recall_3']:.4f}/{t2i_metrics['recall_5']:.4f}/{t2i_metrics['recall_10']:.4f}")
        print(f"I2T Recall@1/3/5/10: {i2t_metrics['recall_1']:.4f}/{i2t_metrics['recall_3']:.4f}/{i2t_metrics['recall_5']:.4f}/{i2t_metrics['recall_10']:.4f}")

        return self.results

    def eval(self, model_name: str, model_list: List, dataset: str = "coco_retrieval", batch_size: int = 32):
        self.results = {
            'model_name': model_name,
            'dataset': dataset,
        }
        self.eval_coco_retrieval(model_name, model_list, dataset, batch_size)
        return self.results, self.prediction_dict
