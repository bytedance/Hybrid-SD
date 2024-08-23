import sys
sys.path.append("/mnt/bn/bytenn-data2/liuhj/pylib")
import collections
import os
import random
from tqdm import tqdm
import re
from torchvision.utils import save_image
import shutil
import  yaml
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import crop
from transformers import CLIPTextModel, CLIPTokenizer
USED_KEYS = {"jpg": "instance_images", "json": "instance_prompt_ids"}


def verify_keysxl(samples, required_keys, handler=wds.handlers.reraise_exception):
    """
    Requires that both the image and embedding are present in the sample
    This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
    """

    for sample in samples:
        try:
            lines = sample["json"]
            sample_json = lines
        except Exception as e:
            print("##############",str(e))
            continue
        import pdb;pdb.set_trace()
        w, h = sample["jpg"].size
        
        if "aesthetic_score" in sample_json:
            aesthetic_score = sample_json["aesthetic_score"]
            if aesthetic_score < 5.65 or w*h<900*900 or w<900 or h<900:
                continue
        if "watermark" in sample_json:
            watermark = sample_json["watermark"]
            if watermark > 0.5:
                continue
        is_normal = True
        for key in required_keys:
            if key not in sample:
                print(f"Sample {sample['__key__']} missing {key}. Has keys {sample.keys()}")
                is_normal = False
        if is_normal:
            yield {key: sample[key] for key in required_keys}

key_verifierxl = wds.filters.pipelinefilter(verify_keysxl)


class ImageEmbeddingDatasetXL(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            handler=wds.handlers.reraise_exception,
            resample=False,
            shuffle_shards=True,
    ):
        super().__init__()
        keys = list(USED_KEYS.keys())
        self.resampling = resample

        if resample:
            assert not shuffle_shards, "Cannot both resample and shuffle"
            self.append(wds.ResampledShards(urls))
        else:
            self.append(wds.SimpleShardList(urls))
            if shuffle_shards:
                self.append(wds.filters.shuffle(1000))

        self.append(wds.tarfile_to_samples(handler=handler))

        self.append(wds.decode("pilrgb", handler=handler))

        self.append(key_verifierxl(required_keys=keys,  handler=handler))

        self.append(wds.map(self.preproc))

    def preproc(self, sample):
        """Applies the preprocessing for images"""
        example = {}
        instance_image = sample["jpg"]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["image"] = instance_image
        
        sample_json = sample["json"]
        example["text"] = sample_json["caption"]
        return example

    # def __len__(self):
    #     # LAION Aesthetics V2 6+
    #     return 8483623
        

def WebDataset(url):
    print(f'loading dataset from path: {url}')
    urls = [os.path.join(url, file_name) for file_name in os.listdir(url) if file_name.endswith('.tar')]
    print(f'load dataset done')

    dataset = ImageEmbeddingDatasetXL(
        urls,
        shuffle_shards=True,
        resample=False,
        handler=wds.handlers.warn_and_continue
    )
    return dataset


if __name__ == "__main__":
    url = "/mnt/bn/bytenn-data2/liuhj/Laion_aesthetics_5plus_1024_33M"
    batch_size = 16
    
    train_dataset = WebDataset(url)
   
    for i, batch in enumerate(train_dataset):
        print(i)
        #print(batch["text"])
        #image = batch["image"]
        #image.save("./test_image/" + str(i)+'.j')