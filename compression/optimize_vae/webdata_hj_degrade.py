"""
training vae only, remove the text tokenize procedure
"""
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
import math
import numpy as np
import torch
from PIL import Image
import cv2
import copy
from torch.utils.data import Dataset
import webdataset as wds
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import crop
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)
from compression.optimize_vae import degradations as degradations
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
        
        if "aesthetic_score" in sample_json:
            aesthetic_score = sample_json["aesthetic_score"]
            if aesthetic_score < 5.65 or w*h<600*700 or w<600 or h<700:
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




#############################
# degradtion from CFPGAN


opt = {}
opt['blur_kernel_size']  =  7
opt['kernel_list'] =  ['iso', 'aniso']
opt['kernel_prob'] = [0.5, 0.5]
opt['blur_sigma'] = [0.1, 1]
opt['noise_range'] = [0, 40]
opt['jpeg_range'] = [60, 100]
opt['color_jitter_prob'] = 0.3
opt['color_jitter_shift'] = 20



def gen_degradation(opt, img):
    ### blur ###
    kernel = degradations.random_mixed_kernels(
        opt['kernel_list'],
        opt['kernel_prob'],
        opt['blur_kernel_size'],
        opt['blur_sigma'],
        opt['blur_sigma'], [-math.pi, math.pi],
        noise_range=None)
    img = cv2.filter2D(img, -1, kernel)
    ### noise ###
    if opt['noise_range'] is not None:
        img = degradations.random_add_gaussian_noise(img, opt['noise_range'])
    # if opt['jpeg_range'] is not None:
    #     img = degradations.random_add_jpg_compression(img, opt['jpeg_range'])
    # if opt['color_jitter_prob'] is not None:
    #     """jitter color: randomly jitter the RGB values, in numpy formats"""
    #     jitter_val = np.random.uniform(-opt['color_jitter_shift'], opt['color_jitter_shift'], 3).astype(np.float32)
    #     img = img + jitter_val
    #     img = np.clip(img, 0, 1)
    return img

"""
TODO: get_component_coordinates
"""

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
            center_crop=True,
            degrade=False,
            opt = opt,
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
        self.opt = opt
        print(self.opt)
        self.degrade = degrade
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

    def gen_degradation(self, opt, img):
        ### blur ###
        kernel = degradations.random_mixed_kernels(
            opt['kernel_list'],
            opt['kernel_prob'],
            opt['blur_kernel_size'],
            opt['blur_sigma'],
            opt['blur_sigma'], [-math.pi, math.pi],
            noise_range=None)
        img_de = cv2.filter2D(img, -1, kernel)

        ### noise ###
        if opt['noise_range'] is not None:
            img_de = degradations.random_add_gaussian_noise(img_de, opt['noise_range'])
        # if opt['jpeg_range'] is not None:
        #     img_de = degradations.random_add_jpg_compression(img_de, opt['jpeg_range'])
        # if opt['color_jitter_prob'] is not None:
        #     """jitter color: randomly jitter the RGB values, in numpy formats"""
        #     jitter_val = np.random.uniform(-opt['color_jitter_shift'], opt['color_jitter_shift'], 3).astype(np.float32)
        #     img = img + jitter_val
        #     img = np.clip(img, 0, 1)
        return img_de

    def preproc(self, sample):
        """Applies the preprocessing for images"""

        example = {}
        instance_image = sample["jpg"]
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["original_size"] = instance_image.size

        # resize
        instance_image = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)(instance_image)
        # PIL Image
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

        # degrade
        if self.degrade:
            img = np.array(instance_image).astype(np.float32) / 255. #[0,1]
            degrade_image = self.gen_degradation(self.opt, img)
            degrade_image = Image.fromarray(np.uint8(degrade_image * 255.))
            example["degrade_image"] = self.image_transforms(degrade_image)

        example["instance_images"] = self.image_transforms(instance_image)
        sample_json = sample["json"]
        example["instance_en"] = sample_json["caption"]
        return example

        

def collate_fn_de(examples):
    pixel_values = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    
    lq = [example["degrade_image"] for example in examples]
    lq = torch.stack(lq)
    lq = lq.to(memory_format=torch.contiguous_format).float()
    
    batch = {
        "pixel_values": pixel_values,
        "lq_values": lq,
    }
    return batch

def collate_fn(examples):
    pixel_values = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values
    }
    return batch


def WebDataset(url, batch_size, size=512,num_workers=1, prefetch_factor=32, degrade=False):
    print(f'loading dataset from path: {url}')
    urls = [os.path.join(url, file_name) for file_name in os.listdir(url) if file_name.endswith('.tar')]
    print(f'load dataset done')


    dataset = ImageEmbeddingDataset(
        urls,
        shuffle_shards=False, #True,
        resample=False,
        size=size,
        handler=wds.handlers.warn_and_continue,
        degrade=degrade,
        opt=opt
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
            collate_fn=collate_fn_de if degrade else collate_fn,
            drop_last=True,
        )
    return loader



if __name__ == "__main__":
    # url = "/mnt/bn/bytenn-data2/liuhj/Laion_aesthetics_5plus_1024_33M"
    url = '/mnt/bn/bytenn-yg2/datasets/Laion_aesthetics_5plus_1024_33M/Laion33m_data_test'
    batch_size = 64
    
    train_dataloader = WebDataset(url, batch_size=batch_size,size=512, degrade = True)

    for i, batch in enumerate(train_dataloader):
        print(i)
        image  = ((batch["pixel_values"] +1) * 127.5).detach().clamp(0, 255).to(torch.uint8).permute(0,2,3,1).numpy()
        image_lq = ((batch["lq_values"] +1) * 127.5).detach().clamp(0, 255).to(torch.uint8).permute(0,2,3,1).numpy()
        print(np.mean(image - image_lq))
        for j in range(len(image)):
            img = Image.fromarray(image[j])
            img.save("/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/test_img/" + "img" + str(i) + '_' + str(j) +'.jpeg')
            
            img_lq = Image.fromarray(image_lq[j])
            img_lq.save("/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/test_img/" + "img" + str(i) + '_' + str(j) +'_lq.jpeg')


        # print(batch["pixel_values"])
        # print(batch["input_ids"])
        #image.save("./test_image/" + str(i)+'.j'