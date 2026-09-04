import json
from .cola import ColaDataset
from .scpp import ScppDataset
from .negbench import NegbenchDataset
from .coco import COCODataset
from .mcmr import MCMRDataset
from .flickr30k import Flickr30kDataset
def load_cola(**kwargs):
    return ColaDataset(**kwargs)

def load_scpp(**kwargs):
    return ScppDataset(**kwargs)

def load_negbench(**kwargs):
    return NegbenchDataset(**kwargs)

def load_coco(**kwargs):
    return COCODataset(**kwargs)

def load_mcmr(**kwargs):
    return MCMRDataset(**kwargs)

def load_flickr30k(**kwargs):
    return Flickr30kDataset(**kwargs)

dataset_dict = {
    "cola": load_cola,
    "scpp" : load_scpp,
    "negbench" : load_negbench,
    "coco" : load_coco,
    "mcmr": load_mcmr,
    "flickr30k": load_flickr30k,
}

def load_dataset(dataset_name: str, **kwargs):
    return dataset_dict[dataset_name](**kwargs)