## Datasets Setup

All benchmarks are organized under a single data root directory, referred to as `$DATA_ROOT` below. After downloading, arrange the files as follows, and point the path constants at the top of each task file in `core/tasks/` to this layout.

```
$DATA_ROOT/
├── datasets/
│   ├── coco2017_val/val2017/                          # COCO 2017 val images
│   └── composite-reason-benchmark/VOC2007_raw/imgs/   # VOC2007 images
└── benchmark/
    ├── COLA/
    │   ├── data/COLA_multiobjects_matching_benchmark_new.json
    │   └── imgs/                                      # COLA images
    ├── scpp/data/                                     # 5 SugarCrepe++ json files
    ├── negbench/data/images/                          # 4 NegBench csv files
    ├── flickr30k_test/
    │   ├── test_1k_flickr.csv
    │   └── images_flickr_1k_test/                     # 1k test images
    └── MCMR-datasets/
        ├── images/                                    # product images
        └── uni_mteb_eval_t2ti/                        # queries/instances/corpus.jsonl
```

### 1 Compositional Reasoning

#### 1.1 COLA

```sh
hf download array/cola
```

Convert `cola/data/multiobjects.parquet` into a json file where each item follows the format below, with the image entries pointing to the downloaded images under `benchmark/COLA/imgs/`:

```
['img0', 'caption0', 'img1', 'caption1']
```

#### 1.2 SugarCrepe++

```sh
hf download Aman-J/SugarCrepe_pp
```

Place the five split files (`replace_att.json`, `replace_obj.json`, `replace_rel.json`, `swap_att.json`, `swap_obj.json`) under `benchmark/scpp/data/`.

For the images, download can be done using both https://cocodataset.org/#download or https://opendatalab.com/OpenDataLab/COCO_2017.

#### 1.3 NegBench

Download the csv files from https://drive.google.com/drive/folders/1kSEq0mkV1t1T8GuOAM65iz_iAA7e5gxB?usp=sharing for evaluation, and place them under `benchmark/negbench/data/images/`.

For the images:
- COCO val2017: download from https://cocodataset.org/#download or https://opendatalab.com/OpenDataLab/COCO_2017.
- VOC2007: download the official archive from http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar and place the JPEG images under `datasets/composite-reason-benchmark/VOC2007_raw/imgs/`.

### 2 General Retrieval

#### 2.1 COCO

For the dataset, download can be done using both https://cocodataset.org/#download or https://opendatalab.com/OpenDataLab/COCO_2017. We select the val2017 split.

#### 2.2 Flickr30k

```sh
hf download royokong/flickr30k_test
```

Unzip the archive and place `test_1k_flickr.csv` and `images_flickr_1k_test/` under `benchmark/flickr30k_test/`.

#### 2.3 MCMR

```sh
hf download Lux1997/MCMR
```

Convert the dataset to the mteb format (`queries.jsonl`, `instances.jsonl`, `corpus.jsonl`) using the scripts under `benchmark/MCMR-datasets/` (`make_candidates.py`, `make_query_and_instances.py`), and update the path constants inside the scripts to your local layout.
