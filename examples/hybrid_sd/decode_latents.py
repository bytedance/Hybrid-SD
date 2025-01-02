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

from compression.optimize_vae.models.autoencoder_tiny import AutoencoderTinyWS, AutoencoderTiny
from compression.optimize_vae.models.autoencoder_kl import AutoencoderKL
from compression.utils.misc import change_img_size
import torch
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# tiny vae
path = ""
chk_path = ""

vae_config = AutoencoderTiny.load_config(path)
vae = AutoencoderTiny.from_config(vae_config).eval()
vae.load_state_dict(torch.load(chk_path), strict=True)

vae = vae.to(torch.float16).cuda()

# decoding loop

pt_paths = ["results/HybridSD_LCM_guidance7_ours_Tiny_scale/SD14_LCM-ours-tiny_lcm-0,8/im512",
           "results/HybridSD_LCM_guidance7_ours_Tiny_scale/SD14_LCM-ours-tiny_lcm-4,4/im512",
           "results/HybridSD_LCM_guidance7_ours_Tiny_scale/SD14_LCM-ours-tiny_lcm-8,0/im512"
           ]

class TensorDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.file_list.sort()  # Ensure consistent ordering

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        tensor = torch.load(file_path)
        return tensor, file_path

def create_dataloader(data_dir, batch_size=128, num_workers=4, shuffle=True):
    dataset = TensorDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      num_workers=num_workers)

for i in range(len(pt_paths)):
    pt_path = pt_paths[i]
    dataloader_pt = create_dataloader(pt_path, batch_size=10, num_workers=0, shuffle=False)
    save_dir =  pt_path.replace('im512', 'image512_ours') 
    os.makedirs(save_dir, exist_ok=True)
    for batch, files in dataloader_pt:
        batch = batch.to(torch.float16).cuda()
        batch = batch / vae.config.scaling_factor
        imgs = vae.decode(batch)['sample']
        
        imgs = ((imgs.detach() + 1.) * 127.5 ).clamp(0, 255).permute(0, 2,3,1).to(torch.uint8).cpu().numpy()
        
        for i in range(len(files)):
            name = files[i].split('/')[-1].split('.')[0]
            Image.fromarray(imgs[i]).save(os.path.join(save_dir, name + '.jpg'))


    target_dir = save_dir.replace('512', '256')
    os.makedirs(target_dir, exist_ok=True)
    change_img_size(save_dir, target_dir, 256)