import sys
sys.path.append("/mnt/bn/bytenn-yg2/liuhj/pylib")
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
        
        w, h = sample["jpg"].size   
        if  w*h<600*700 or w<512 or h<512:
            continue
        if "pwatermark" in sample_json:
            watermark = sample_json["pwatermark"]
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


class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            tokenizer=None,
            extra_keys=[],
            size= 512,
            handler=wds.handlers.reraise_exception,
            resample=False,
            shuffle_shards=True,
            center_crop=True
    ):
        super().__init__()
        keys = list(USED_KEYS.keys()) + extra_keys
        # self.key_map = {key: i for i, key in enumerate(keys)}
        self.resampling = resample
        self.center_crop = center_crop
        self.size = size
        self.crop = transforms.CenterCrop(size) if center_crop else transforms.Lambda(crop_left_upper)
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.tokenizer = tokenize_captions

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

        example["original_size"] = instance_image.size

        # resize
        instance_image = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)(instance_image)
        # crop
        if self.center_crop:
            w,h = instance_image.size
            instance_image = self.crop(instance_image)
            if min(w,h)>self.size:
                example["crops_coords_top_left"] = (int(h/2-self.size),int(w/2-self.size))
            else:
                example["crops_coords_top_left"] = (int(h/2-min(w,h)/2),int(w/2-min(w,h)/2))
        else:
            example["crops_coords_top_left"],instance_image = self.crop(instance_image)

        example["instance_images"] = self.image_transforms(instance_image)
        
        sample_json = sample["json"]
        example["instance_en"] = sample_json["caption"]
        return example

        

tokenizer = CLIPTokenizer.from_pretrained('/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5', subfolder="tokenizer")

def tokenize_captions(texts, is_train=True):
        inputs = tokenizer(texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids

def collate_fn(examples):
    # instance_prompt_ids = [example["instance_prompt_ids"] for example in examples]
    texts_en = [example["instance_en"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = tokenize_captions(texts_en)
    batch = {
        "pixel_values": pixel_values,
        "input_ids": input_ids
    }

    return batch


def WebDatasetAE(url, batch_size, size=512,num_workers=8, prefetch_factor=32):
    print(f'loading dataset from path: {url}')
    urls = [os.path.join(url, file_name) for file_name in os.listdir(url) if file_name.endswith('.tar')]
    print(f'load dataset done')

    dataset = ImageEmbeddingDataset(
        urls,
        shuffle_shards=True,
        resample=False,
        size=size,
        handler=wds.handlers.warn_and_continue
    )
    from prefetch_generator import BackgroundGenerator
    class DataLoaderX(DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())
    
    loader = DataLoaderX(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,  # This might be good to have high so the next npy file is prefetched
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
        )
    return loader


def freq_crop(input_t, band_width):

    img_size = input_t.size(-1)
    band_w = band_width // 2

    img_f = torch.fft.fft2(input_t)

    img_crop = torch.empty(
        [input_t.size(0), input_t.size(1), band_w * 2, band_w * 2],
        dtype=img_f.dtype, device=img_f.device
        )

    img_crop[:, :, :band_w, :band_w] = img_f[:, :, :band_w, :band_w]
    img_crop[:, :, -band_w:, :band_w] = img_f[:, :, -band_w:, :band_w]
    img_crop[:, :, :band_w, -band_w:] = img_f[:, :, :band_w, -band_w:]
    img_crop[:, :, -band_w:, -band_w:] = img_f[:, :, -band_w:, -band_w:]

    img_crop = img_crop * ((band_w * 2 / img_size) ** 2)

    img_crop = torch.fft.ifft2(img_crop)
    img_crop = torch.real(img_crop)

    return img_crop

if __name__ == "__main__":
    # laion AE
    url = "/mnt/bn/bytenn-yg2/datasets/laion2b_en_aesthetics/data" 
    #url = "/mnt/bn/bytenn-yg2/datasets/cc12m/data"
    #url = '/mnt/bn/bytenn-yg2/datasets/Laion_aesthetics_5plus_1024_33M/Laion33m_data'
    batch_size = 32
    train_dataloader = WebDatasetAE(url, batch_size=batch_size,size=512, num_workers=8)
   
    for i, batch in enumerate(train_dataloader):
        pass
        #print(i)
        image  = ((batch["pixel_values"] +1) * 127.5).detach().clamp(0, 255).to(torch.uint8).permute(0,2,3,1).numpy()
        for j in range(len(image)):
            img = Image.fromarray(image[j])
            img.save("/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/test_img/" + "img" + str(i) + '_' + str(j) +'.jpeg')

        # print(batch["pixel_values"])
        # print(batch["input_ids"])
