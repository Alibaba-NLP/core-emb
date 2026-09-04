import time
import torch
from torch.nn import functional as F
from embedding import i2t_emb, fused_emb
from typing import List
from tqdm import tqdm

class CompDataset():

    def __init__(self, **kwargs):
        
        self.mrl_dim = kwargs.get("mrl_dim", -1)

    @torch.no_grad()
    def img_emb(self, model_name: str, model_list: List, img_paths: List[str]):
        
        img_embs = i2t_emb(model_name=model_name, model_list=model_list, texts=img_paths, is_query=False)

        if self.mrl_dim > 0:
            img_embs = img_embs[:,:self.mrl_dim] 
        img_embs = F.normalize(img_embs,dim=-1)

        # validation
        if self.mrl_dim > 0:
            assert img_embs.shape[1] == self.mrl_dim
            assert img_embs.shape[0] == len(img_paths)
        else:
            assert img_embs.shape[0] == len(img_paths)
        return img_embs 

    @torch.no_grad()
    def text_emb(self, model_name: str, model_list: List, texts: List[str]):
        
        text_embs = i2t_emb(model_name=model_name, model_list=model_list, texts=texts, is_query=True)
        if self.mrl_dim > 0:
            text_embs = text_embs[:,:self.mrl_dim] 
            
        text_embs = F.normalize(text_embs,dim=-1)

        # validation
        if self.mrl_dim > 0:
            assert text_embs.shape[1] == self.mrl_dim
            assert text_embs.shape[0] == len(texts)
        else:
            assert text_embs.shape[0] == len(texts)

        return text_embs

    @torch.no_grad()
    def img_batch_emb(self, model_name: str, model_list: List, img_paths: List[str], batch_size: int =32):
        start = time.time()
        print("Start img batch embedding")
        img_embs = []
        for i in tqdm(range(0, len(img_paths), batch_size)):
            tmp_img_embs = i2t_emb(model_name=model_name, model_list=model_list, texts=img_paths[i:i+batch_size], is_query=False)
            img_embs.append(tmp_img_embs)

        img_embs = torch.cat(img_embs,dim=0)
        if self.mrl_dim > 0:
            img_embs = img_embs[:,:self.mrl_dim] 
        img_embs = F.normalize(img_embs,dim=-1)

        # validation
        if self.mrl_dim > 0:
            assert img_embs.shape[1] == self.mrl_dim
            assert img_embs.shape[0] == len(img_paths)
        else:
            assert img_embs.shape[0] == len(img_paths)

        end = time.time()
        print("Finish img batch embedding, time: {:.4f}".format(end-start))
        return img_embs 

    @torch.no_grad()
    def text_batch_emb(self, model_name: str, model_list: List, texts: List[str], batch_size: int =32):
        start = time.time()
        text_embs = []
        for i in tqdm(range(0, len(texts), batch_size)):
            tmp_text_embs = i2t_emb(model_name=model_name, model_list=model_list, texts=texts[i:i+batch_size], is_query=True)
            text_embs.append(tmp_text_embs)
        text_embs = torch.cat(text_embs,dim=0)

        if self.mrl_dim > 0:
            text_embs = text_embs[:,:self.mrl_dim]
        
        # validation
        if self.mrl_dim > 0:
            assert text_embs.shape[1] == self.mrl_dim
            assert text_embs.shape[0] == len(texts)
        else:
            assert text_embs.shape[0] == len(texts)

        text_embs = F.normalize(text_embs,dim=-1)
        end = time.time()
        print("Finish text batch embedding, time: {:.4f}".format(end-start))
        return text_embs

    @torch.no_grad()
    def fused_batch_emb(self, model_name: str, model_list: List, texts: List[str], images: List[str], batch_size: int = 32):
        """
        Generate fused embeddings from both text and image inputs in batches.

        Args:
            model_name: Name of the model
            model_list: List containing model and processor
            texts: List of text strings
            images: List of image paths (same length as texts)
            batch_size: Batch size for processing

        Returns:
            Fused embeddings normalized to unit length
        """
        assert len(texts) == len(images), f"texts and images must have same length, got {len(texts)} and {len(images)}"

        start = time.time()
        print("Start fused batch embedding")
        fused_embs = []
        for i in tqdm(range(0, len(texts), batch_size)):
            tmp_fused_embs = fused_emb(
                model_name=model_name,
                model_list=model_list,
                texts=texts[i:i+batch_size],
                images=images[i:i+batch_size]
            ).detach()
            fused_embs.append(tmp_fused_embs)

        fused_embs = torch.cat(fused_embs, dim=0)

        if self.mrl_dim > 0:
            fused_embs = fused_embs[:, :self.mrl_dim]

        fused_embs = F.normalize(fused_embs, dim=-1)

        # validation
        if self.mrl_dim > 0:
            assert fused_embs.shape[1] == self.mrl_dim
            assert fused_embs.shape[0] == len(texts)
        else:
            assert fused_embs.shape[0] == len(texts)

        end = time.time()
        print("Finish fused batch embedding, time: {:.4f}".format(end - start))
        return fused_embs