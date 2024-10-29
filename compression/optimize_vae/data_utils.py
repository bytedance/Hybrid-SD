# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure
from torchmetrics.aggregation import MeanMetric
from torchvision.io import read_image
from cleanfid import fid
from torch.utils.data import DataLoader, Dataset
import os
from tqdm.auto import tqdm

import math
import os
import random
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.optim import RAdam
import torchvision.transforms.functional  as TF
import transformers
from PIL import Image
from torchvision import transforms



class MultiImageDataset(Dataset):
    def __init__(self, root0, root1, is_gt=False):
        super().__init__()
        self.root0 = root0
        self.root1 = root1
        file_names0 = os.listdir(root0)
        file_names1 = os.listdir(root1)

        self.image_names0 = sorted([name for name in file_names0 if name.endswith(".png") or name.endswith(".jpg")])
        self.image_names1 = sorted([name for name in file_names1 if name.endswith(".png") or name.endswith(".jpg")])
        self.is_gt = is_gt
        assert len(self.image_names0) == len(self.image_names1)

    def __len__(self):
        return len(self.image_names0)

    def __getitem__(self, idx):
        img0 = read_image(os.path.join(self.root0, self.image_names0[idx]))
        img1 = read_image(os.path.join(self.root1, self.image_names1[idx]))
         # img0 = np.array(Image.open(os.path.join(self.root0, self.image_names0[idx])).convert("RGB"))
        # img1 = np.array(Image.open(os.path.join(self.root1, self.image_names1[idx])).convert("RGB"))
        batch_list = [img0, img1]
        return batch_list


def evaluation_coco2017(vae,real_path,out_path,weight_dtype,logger):
    psnr_metric = []
    lpips_metric = []
    weight_sum = 0
    psnr = PeakSignalNoiseRatio().to(vae.device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(vae.device)
    dataset = MultiImageDataset(real_path, out_path)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=8) #collate_fn=my_collate
    progress_bar = tqdm(dataloader)
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            batch = [img.to(vae.device) for img in batch]
            batch_size = batch[0].shape[0]
            psnr_metric.append(psnr(batch[0].to(weight_dtype), batch[1].to(weight_dtype)).item() * batch_size)
            lpips_metric.append(lpips(batch[0]/255, batch[1]/255).item() * batch_size)
            weight_sum += batch_size
            
    psnr_score = np.sum(psnr_metric)/weight_sum
    lpips_score = np.sum(lpips_metric)/weight_sum
    try:
        fid_score = fid.compute_fid(real_path, out_path)
    except:
        fid_score = 10000
    return psnr_score,lpips_score,fid_score
    

def load_visualization_imgs(image_dir="datasets/eval_img"):
    """ data range [-1,1] """
    transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]
    visual_imgs = []
    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")
        image = transform(image).unsqueeze(0)
        visual_imgs.append(image)
    return torch.cat(visual_imgs, dim=0)


if __name__ == "__main__":
    res = load_visualization_imgs(image_dir="datasets/eval_img")

