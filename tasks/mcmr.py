from typing import List
import os
import json
import torch
from tqdm import tqdm
from .composite_dataset import CompDataset
import pytrec_eval
from config import MCMR_DATASET_PATHS


class MCMRDataset(CompDataset):
    """
    MCMR (Multi-Modal Content Retrieval) dataset for text-to-multi-modal retrieval.
    Evaluates retrieval models using pytrec_eval.
    Metrics: MAP, NDCG, Recall.
    """

    def __init__(self, dataset_name: str = "MCMR_T2TI", **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.queries_dict = {}  # qid -> query_text
        self.qrel = {}  # qid -> {cid: 1}
        self.corpus_dict = {}  # cid -> {"id", "text", "image", "modality"}
        self.results = {}
        self.prediction_dict = {}
        self.load()

    def load(self):
        """Load MCMR dataset from jsonl files."""
        if self.dataset_name not in MCMR_DATASET_PATHS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}. Available: {list(MCMR_DATASET_PATHS.keys())}")

        dataset_path = MCMR_DATASET_PATHS[self.dataset_name]

        # Load queries
        queries_path = os.path.join(dataset_path, "queries.jsonl")
        with open(queries_path, "r") as f:
            queries = [json.loads(line) for line in f]
        for query in queries:
            self.queries_dict[query["id"]] = query["text"]

        # Load instances (qrel)
        instances_path = os.path.join(dataset_path, "instances.jsonl")
        with open(instances_path, "r") as f:
            instances = [json.loads(line) for line in f]
        for instance in instances:
            qid = instance["qid"]
            if qid not in self.qrel:
                self.qrel[qid] = {}
            for pos_id in instance["pos"]:
                self.qrel[qid][pos_id] = 1

        # Load corpus
        corpus_path = os.path.join(dataset_path, "corpus.jsonl")
        with open(corpus_path, "r") as f:
            corpus = [json.loads(line) for line in f]
        for item in corpus:
            self.corpus_dict[item["id"]] = item

        print(f"Loaded MCMR dataset '{self.dataset_name}':")
        print(f"  - Queries: {len(self.queries_dict)}")
        print(f"  - Corpus: {len(self.corpus_dict)}")
        print(f"  - Qrel pairs: {sum(len(v) for v in self.qrel.values())}")

    def eval_mcmr_retrieval(self, model_name: str, model_list: List, batch_size: int = 32, sim_batch_size: int = 1000):
        """
        Evaluate text-to-multi-modal retrieval on MCMR dataset.
        Queries are text-only, corpus items are multi-modal (image + text).
        Uses memory-efficient batched similarity computation.
        """
        # Prepare query texts and corpus data
        query_ids = list(self.queries_dict.keys())
        query_texts = [self.queries_dict[qid] for qid in query_ids]

        corpus_ids = list(self.corpus_dict.keys())
        corpus_images = [self.corpus_dict[cid]["image"] for cid in corpus_ids]
        corpus_texts = [self.corpus_dict[cid]["text"] for cid in corpus_ids]

        # This code block is simple for test
        # query_ids = query_ids[:10]
        # query_texts = query_texts[:10]
        # corpus_ids = []
        # for qid in query_ids:
        #     rel = self.qrel[qid]
        #     corpus_ids += (rel.keys())

        # corpus_images = [self.corpus_dict[cid]["image"] for cid in corpus_ids]
        # corpus_texts = [self.corpus_dict[cid]["text"] for cid in corpus_ids]

        # Get embeddings
        print("Encoding queries...")
        query_embs = self.text_batch_emb(
            model_name=model_name,
            model_list=model_list,
            texts=query_texts,
            batch_size=batch_size
        ).detach()

        print("Encoding corpus (fused image + text)...")
        corpus_embs = self.fused_batch_emb(
            model_name=model_name,
            model_list=model_list,
            texts=corpus_texts,
            images=corpus_images,
            batch_size=batch_size
        ).detach()

        num_queries = len(query_ids)
        num_corpus = len(corpus_ids)

        # Build predictions dict for pytrec_eval with batched computation
        predictions = {}
        for query_start in tqdm(range(0, num_queries, sim_batch_size), desc="Matrix computation"):
            query_end = min(query_start + sim_batch_size, num_queries)
            query_batch = query_embs[query_start:query_end]
            # move corpus_embs to gpu

            # Compute similarities for this batch
            sims_batch = query_batch @ corpus_embs.T  # (batch_size, num_corpus)

            for local_idx, q_idx in enumerate(range(query_start, query_end)):
                sim_scores = sims_batch[local_idx]
                # Only keep top-100 scores to reduce memory
                top_k = min(100, num_corpus)
                top_indices = torch.topk(sim_scores, top_k).indices.tolist()
                pred = {str(corpus_ids[idx]): float(sim_scores[idx]) for idx in top_indices}
                predictions[str(query_ids[q_idx])] = pred

            del sims_batch

        # Evaluate using pytrec_eval
        evaluator = pytrec_eval.RelevanceEvaluator(
            self.qrel,
            {"map_cut", "ndcg_cut", 
             "recall_1", "recall_3", "recall_5", "recall_10",
              "recall_50", "recall_100"}
        )
        eval_results = evaluator.evaluate(predictions)
        # breakpoint()
        metrics = {}
        for qid, result in eval_results.items():
            for metric, val in result.items():
                if metric not in metrics:
                    metrics[metric] = []
                metrics[metric].append(val)
        
        for metric, vals in metrics.items():
            metrics[metric] = sum(vals) / len(vals)
        # Store results
        self.results = metrics
        self.prediction_dict = predictions
        
        print(f"Evaluation of MCMR '{self.dataset_name}' on {model_name} done.")
        print(f"Recall@1/3/5/10: {metrics['recall_1']:.4f}/{metrics['recall_3']:.4f}/{metrics['recall_5']:.4f}/{metrics['recall_10']:.4f}")
        print(f"NDCG@5/10:/{metrics['ndcg_cut_5']:.4f}/{metrics['ndcg_cut_10']:.4f}")
        print(f"MAP@10: {metrics['map_cut_10']:.4f}")
        return self.results

    def eval(self, model_name: str, model_list: List, dataset: str = None, batch_size: int = 32):
        """
        Main evaluation function.

        Args:
            model_name: Name of the model
            model_list: List containing model and processor
            dataset: Dataset name (uses self.dataset_name if None)
            batch_size: Batch size for encoding

        Returns:
            Tuple of (results dict, predictions dict)
        """
        if dataset is None:
            dataset = self.dataset_name

        self.results = {
            'model_name': model_name,
            'dataset': dataset,
        }
        self.eval_mcmr_retrieval(model_name, model_list, batch_size)
        return self.results, self.prediction_dict