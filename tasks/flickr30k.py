from typing import List
import os
import json
import torch
from tqdm import tqdm
from .composite_dataset import CompDataset
import pandas as pd
import pytrec_eval
from config import FLICKR30K_IMAGE_PATH, FLICKR30K_CSV_PATH


class Flickr30kDataset(CompDataset):
    """
    Flickr30k retrieval dataset for text-to-image and image-to-text retrieval.
    Standard evaluation uses the official test set (1000 images, 5 captions each).
    Metrics: Recall@1/3/5/10, NDCG, MAP.
    """

    FLICKR_IMAGE_PATH = FLICKR30K_IMAGE_PATH
    FLICKR_CSV_PATH = FLICKR30K_CSV_PATH

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flickr_data = []
        self.prediction_dict = {}
        self.results = {}
        self.load()

    def load(self):
        """Load Flickr30k test set from CSV."""
        df = pd.read_csv(self.FLICKR_CSV_PATH)

        for _, row in df.iterrows():
            # Parse the raw captions (JSON array stored as string)
            captions = json.loads(row['raw']) if isinstance(row['raw'], str) else row['raw']
            # imgid is a list with 5 repeated values, take the first one
            img_id_val = json.loads(row['imgid']) if isinstance(row['imgid'], str) else row['imgid']
            img_id = str(img_id_val[0]) if isinstance(img_id_val, list) else str(img_id_val)
            self.flickr_data.append({
                "img_id": img_id,
                "img_path": os.path.join(self.FLICKR_IMAGE_PATH, row['filename']),
                "captions": captions
            })

        print(f"Loaded Flickr30k test set with {len(self.flickr_data)} images...")

    def eval_flickr_retrieval(self, model_name: str, model_list: List, dataset: str = "flickr30k_retrieval",
                               batch_size: int = 32, sim_batch_size: int = 1000):
        """
        Evaluates Flickr30k retrieval with multiple captions per image.
        Supports both text-to-image and image-to-text retrieval.
        Uses memory-efficient batched similarity computation.

        Args:
            model_name: Name of the model being evaluated
            model_list: List containing model and processor
            dataset: Dataset name for logging
            batch_size: Batch size for embedding computation
            sim_batch_size: Batch size for similarity matrix computation
        """
        to_eval_data = self.flickr_data

        imgs, texts, caption_counts, img_ids = [], [], [], []
        qrels_t2i = {}  # text-to-image: captions are queries, images are documents
        qrels_i2t = {}  # image-to-text: images are queries, captions are documents

        # Build qrels and collect data
        text_idx = 0
        for cid, item in enumerate(to_eval_data):
            imgs.append(item['img_path'])
            img_id = item['img_id']
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

        # Compute embeddings
        img_embs = self.img_batch_emb(model_name=model_name, model_list=model_list,
                                        img_paths=imgs, batch_size=batch_size).detach()
        text_embs = self.text_batch_emb(model_name=model_name, model_list=model_list,
                                         texts=texts, batch_size=batch_size).detach()

        num_imgs = len(imgs)
        num_texts = len(texts)

        # Text-to-Image retrieval with batched computation
        t2i_predictions = {}
        for text_start in tqdm(range(0, num_texts, sim_batch_size), desc="T2I Matrix computation"):
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

        print(f"Evaluation of Flickr30k retrieval on {model_name} done.")
        print(f"T2I Recall@1/3/5/10: {t2i_metrics['recall_1']:.4f}/{t2i_metrics['recall_3']:.4f}/{t2i_metrics['recall_5']:.4f}/{t2i_metrics['recall_10']:.4f}")
        print(f"I2T Recall@1/3/5/10: {i2t_metrics['recall_1']:.4f}/{i2t_metrics['recall_3']:.4f}/{i2t_metrics['recall_5']:.4f}/{i2t_metrics['recall_10']:.4f}")

        return self.results

    def eval(self, model_name: str, model_list: List, dataset: str = "flickr30k_retrieval", batch_size: int = 32):
        """Main evaluation entry point."""
        self.results = {
            'model_name': model_name,
            'dataset': dataset,
        }
        self.eval_flickr_retrieval(model_name, model_list, dataset, batch_size)
        return self.results, self.prediction_dict