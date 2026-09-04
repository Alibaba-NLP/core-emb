<h1 align="center">CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.04083"><img src="https://img.shields.io/badge/arXiv-CORE-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/collections/Alibaba-NLP/core-emb"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC--BY--4.0-blue.svg" alt="License: CC-BY-4.0"></a>
</p>

Official evaluation code for **CORE**, a framework that transfers compositional reasoning from a multimodal reranker to an MLLM embedding model via listwise rank distillation (Rank-KL).

## Overview

MLLM-based embedding models still fail at compositional retrieval, often ranking scenes with the same concepts but different attribute-object bindings (e.g., "a white plate and a black chair" vs. "a black plate and a white chair") as equally similar. The same backbone, however, resolves these distinctions when used as a cross-attentive reranker. CORE closes this gap:

- **Five-level compositional matching taxonomy**: full match, partial presence, attribute error, object error, full mismatch — a graded spectrum of compositional similarity instead of binary positive/negative labels.
- **Compositional data synthesis**: graded candidate lists synthesized from LAION-400M seed images with Qwen3-VL-32B (structured scene extraction + query/caption generation) and Z-Image-Turbo (image generation), filtered by automated MLLM verification.
- **Rank-KL distillation**: the embedding student is trained to reproduce the reranker teacher's fine-grained ranking over the full candidate list, preserving partial-match ordering that InfoNCE collapses into a uniform negative class.

**Models**: core-reranker-2b/8b (fine-tuned from Qwen3VL-Reranker) and core-emb-2b/8b (distilled from Qwen3VL-Reranker into VL-Emb).

## Results

- core-reranker-8b reaches an 82.7% total average on COLA, SugarCrepe++, and NegBench, outperforming Jina-Reranker by 10.7 points, while recovering negation sensitivity that standard reranker fine-tuning erodes.
- core-emb-8b achieves the best total average (0.666) among all evaluated embedding models, +5.7 points over its VL-Emb-8B backbone.
- Gains transfer to MCMR (R@1 0.375 → 0.412) without sacrificing COCO and Flickr30K retrieval performance.

## Repository Structure

```
core/
├── main.py               # Evaluation entry point
├── config.py             # Dataset and checkpoint path configuration
├── embedding.py          # Unified embedding interface for all models
├── datasets.md           # Dataset download and setup guide
├── tasks/                # Benchmark implementations
│   ├── dataset.py        # Dataset registry
│   ├── composite_dataset.py  # Base class: image/text/fused embedding helpers
│   ├── cola.py           # COLA
│   ├── scpp.py           # SugarCrepe++
│   ├── negbench.py       # NegBench (VOC/COCO MCQ + negated retrieval)
│   ├── coco.py           # COCO image-text retrieval
│   ├── flickr30k.py      # Flickr30k image-text retrieval
│   └── mcmr.py           # MCMR text-to-multimodal retrieval
└── models/               # Model wrappers
    ├── load_models.py    # Model loading dispatch
    ├── qwen3_vl/         # Qwen3-VL embedding wrapper
    ├── qwen2_5_vl/
    ├── qwen2_vl/
    ├── e5_v/
    ├── reranker/         # Generative reranker
    └── rzen/
```

## Installation

Requires Python 3.10+ and a CUDA device.

```bash
pip install torch transformers pandas pytrec_eval tqdm pillow qwen-vl-utils sentencepiece
```

For reranker evaluation, additionally install [ms-swift](https://github.com/modelscope/ms-swift):

```bash
pip install ms-swift
```

For faster inference on supported GPUs, install FlashAttention separately.

## Dataset Setup

All benchmarks are organized under a single data root directory. Refer to [datasets.md](datasets.md) for download sources and the expected directory layout. In short:

```bash
export DATA_ROOT=/path/to/data   # contains datasets/ and benchmark/
```

## Evaluation

### Quick Start

```bash
cd core
export DATA_ROOT=/path/to/data

python main.py \
    --model_name core-emb-8b \
    --model_path /path/to/core-emb-8b \
    --dataset cola \
    --model_type embed \
    --device cuda:0
```

### Arguments

| Argument | Values | Description |
|---|---|---|
| `--model_name` | e.g. `core-emb`, `core-reranker`, `qwen3vl`, `qwen25vl`, `gme`, `unime`, `vlm2vec`, `umarvel-qwen3vl-4b`, `siglip`, `tripletclip`, `negclip`, `e5-v`, `mme5`, `rzen` | Model family; selects the loading path in `models/load_models.py` |
| `--model_path` | path or HF repo id | Checkpoint path |
| `--dataset` | `cola`, `scpp`, `negbench`, `mcmr`, `coco`, `flickr30k` | Benchmark to evaluate |
| `--model_type` | `embed`, `reranker` | Evaluate as embedding model or generative reranker |
| `--mrl_dim` | int, default `-1` | Truncate embeddings to the first `mrl_dim` dimensions (Matryoshka); `-1` disables |
| `--device` | e.g. `cuda:0` | Device |
| `--output_path` | path | Output root; defaults to `./outputs` |

### Outputs

Results and per-sample predictions are written under the output root:

```
outputs/
├── results/<model>_<dataset>.json                     # embedding model scores
├── predictions/<model>/<dataset>_predictions.json
├── emb_results/<model>_<dataset>.json                 # when --model_type embed (default)
├── emb_predictions/<model>/<dataset>_predictions.json
├── reranker_results/<model>_<dataset>.json            # when --model_type reranker
└── reranker_predictions/<model>/<dataset>_predictions.json
```

Completed runs are skipped automatically, so a failed sweep can simply be re-run.

### Benchmarks

| Benchmark | Task | Metric |
|---|---|---|
| COLA | Two-image two-caption compositional matching | Accuracy |
| SugarCrepe++ | Positive vs. negative caption ranking (5 subsets) | Accuracy |
| NegBench | VOC/COCO MCQ + negated caption retrieval | Accuracy / Recall@5 |
| COCO | Image-text retrieval (val2017) | Recall@1/3/5/10, NDCG, MAP |
| Flickr30k | Image-text retrieval (1K test set) | Recall@1/3/5/10, NDCG, MAP |
| MCMR | Text query to image+text product retrieval | Recall, NDCG, MAP |

### Examples

Evaluate the reranker:

```bash
python main.py --model_name core-reranker-8b --model_path /path/to/core-reranker-8b \
    --dataset negbench --model_type reranker
```

Evaluate with Matryoshka truncation at 1024 dimensions:

```bash
python main.py --model_name core-emb-8b --model_path /path/to/core-emb-8b \
    --dataset scpp --mrl_dim 1024
```

## Citation

If you find this work useful, please cite:

```bibtex
@misc{song2026core,
      title={CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation}, 
      author={Tingyu Song and Mingxin Li and Yanzhao Zhang and Dingkun Long and Chu Liu and Pengjun Xie and Yilun Zhao and Shu Wu},
      year={2026},
      eprint={2609.04083},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.04083}, 
}
```

The citation entry will be updated when the final paper metadata is available.

## Acknowledgements

We thank the authors of [COLA](https://arxiv.org/abs/2305.03689), [SugarCrepe++](https://arxiv.org/abs/2406.11171), and [NegBench](https://github.com/m1k2zoo/negbench) for their benchmarks, and the [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) project for the evaluation framework.
