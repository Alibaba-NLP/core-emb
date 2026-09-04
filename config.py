import os

# Root directory holding all datasets and benchmarks (see datasets.md for the layout).
DATA_ROOT = os.environ.get("DATA_ROOT", "./data")

DATASETS_DIR = os.path.join(DATA_ROOT, "datasets")
BENCHMARK_DIR = os.path.join(DATA_ROOT, "benchmark")

# Raw image datasets
COCO_VAL2017_PATH = os.path.join(DATASETS_DIR, "coco2017_val", "val2017")
VOC2007_PATH = os.path.join(DATASETS_DIR, "composite-reason-benchmark", "VOC2007_raw", "imgs")

# Benchmarks
COLA_DIR = os.path.join(BENCHMARK_DIR, "COLA")
COLA_BENCHMARK_PATH = os.path.join(COLA_DIR, "data", "COLA_multiobjects_matching_benchmark_new.json")
SCPP_DATA_PATH = os.path.join(BENCHMARK_DIR, "scpp", "data")
NEGBENCH_DATA_PATH = os.path.join(BENCHMARK_DIR, "negbench", "data", "images")
FLICKR30K_DIR = os.path.join(BENCHMARK_DIR, "flickr30k_test")
FLICKR30K_IMAGE_PATH = os.path.join(FLICKR30K_DIR, "images_flickr_1k_test")
FLICKR30K_CSV_PATH = os.path.join(FLICKR30K_DIR, "test_1k_flickr.csv")
MCMR_DIR = os.path.join(BENCHMARK_DIR, "MCMR-datasets")
MCMR_DATASET_PATHS = {
    "MCMR_T2TI": os.path.join(MCMR_DIR, "uni_mteb_eval_t2ti"),
}

# Pretrained model checkpoints, organized under one directory (one subdirectory per model).
MODEL_ROOT = os.environ.get("MODEL_ROOT", "./checkpoints")
