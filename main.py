import json


from tqdm import tqdm
import os
from models.load_models import load_model
from tasks.dataset import load_dataset 
from typing import List 
from PIL import Image
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["IMAGE_MAX_TOKEN_NUM"] ="1200"

def main(args):
    # Use output_path if provided, otherwise use default
    if args.output_path:
        base_path = args.output_path.rstrip('/')
        prediction_path = f"{base_path}/predictions/{args.model_name}/{args.dataset}_predictions.json" if args.mrl_dim < 0 else f"{base_path}/predictions_{args.mrl_dim}/{args.model_name}/{args.dataset}_predictions.json"
        result_path = f"{base_path}/results/{args.model_name}_{args.dataset}.json" if args.mrl_dim < 0 else f"{base_path}/results_{args.mrl_dim}/{args.model_name}_{args.dataset}.json"
    else:
        prediction_path = f"outputs/predictions/{args.model_name}/{args.dataset}_predictions.json" if args.mrl_dim < 0 else f"outputs/predictions_{args.mrl_dim}/{args.model_name}/{args.dataset}_predictions.json"
        result_path = f"outputs/results/{args.model_name}_{args.dataset}.json" if args.mrl_dim < 0 else f"outputs/results_{args.mrl_dim}/{args.model_name}_{args.dataset}.json"

    if args.model_type=='embed' :
        result_path = 'emb_' + result_path
        prediction_path = 'emb_' + prediction_path
    elif args.model_type=='reranker' :
        result_path = 'reranker_' + result_path
        prediction_path = 'reranker_' + prediction_path
    else:
        raise ValueError("Invalid model type")

    if os.path.exists(prediction_path) and os.path.exists(result_path):
        print(f"Predictions and results for {args.model_name} on {args.dataset} already exist. ")
        return

    model_list = load_model(model_name=args.model_name, model_path=args.model_path, model_type=args.model_type, device=args.device)
    dataset = load_dataset(args.dataset, mrl_dim=args.mrl_dim)
    
    if args.model_type == 'reranker': 
        results, predictions = dataset.eval_reranker(model_name=args.model_name, model_list=model_list, dataset=args.dataset)
    else: 
        batch_size = 64
        if "mme5" in args.model_name:
            batch_size = 4
        
        results, predictions = dataset.eval(model_name=args.model_name, model_list=model_list, dataset=args.dataset, batch_size=batch_size)

    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    with open(prediction_path, 'w') as f:
        json.dump(predictions, f, indent=4)
        
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation of {args.model_name} on {args.dataset} done. ") 

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="rzen")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="cola", choices=["cola", "scpp", "negbench", "mcmr", "coco", "flickr30k"])
    parser.add_argument("--model_path", type=str, default=None, required=False)
    parser.add_argument("--model_type", type=str, default="embed", choices=["embed", "reranker"])
    parser.add_argument("--mrl_dim", type=int, default=-1, required=False)
    parser.add_argument("--output_path", type=str, default=None, required=False)
    args = parser.parse_args()
    main(args)