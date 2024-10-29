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

import torch 
import torch.nn as nn
from collections import defaultdict
import math
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.attention import BasicTransformerBlock, FeedForward
from copy import deepcopy
import numpy as np
import random

def select_weights_from_index(module, index, op_type='fc', dim=0):
    module.weight = torch.nn.Parameter(torch.index_select(module.weight, dim, index))
    n_channels = len(index)
    if op_type == 'fc':
        assert module.weight.dim() == 2
        if dim == 0:
            module.out_features = n_channels
        elif dim == 1:
            module.in_features = n_channels
        else:
            raise NotImplementedError
    elif op_type == 'conv':
        assert module.weight.dim() == 4
        if dim == 0:
            module.out_channels = n_channels
        elif dim == 1:
            module.in_channels = n_channels
    elif op_type == 'norm':
        module.num_channels = n_channels
    else:
        raise NotImplementedError

    if module.bias is not None and dim == 0:
        module.bias = torch.nn.Parameter(torch.index_select(module.bias, 0, index))


def padding_to_multiple(number, multiple=32):
    return math.ceil(number / multiple) * multiple


def get_select_index(weight, n_origin_channels, n_pruned_channels, device):
    p = 2
    weight_vec = weight.view(n_origin_channels, -1)
    norm = torch.norm(weight_vec, p, 1)
    norm_np = norm.cpu().detach().numpy()
    filter_large_index = torch.from_numpy(norm_np.argsort()[n_pruned_channels:]).to(device)
    selected_index, indices = torch.sort(filter_large_index)
    return selected_index


def do_resnet_prune(module, device, keep_ratio=0.5, min_channels=32):
    num_groups = 32

    print(f"origin layer: {module}")
    # calculate importance score based on conv1.weight
    out_channels = module.conv1.out_channels
    mid_channels = padding_to_multiple(int(out_channels * keep_ratio), num_groups)
    if mid_channels <= min_channels:
        print("meet minimum channels, not prune this resnet block")
        return False

    n_pruned_channels = out_channels - mid_channels

    selected_index = get_select_index(module.conv1.weight, out_channels, n_pruned_channels, device)
    select_weights_from_index(module.conv1, selected_index, 'conv')
    select_weights_from_index(module.time_emb_proj, selected_index, 'fc')
    select_weights_from_index(module.norm2, selected_index, 'norm')

    module.conv2.in_channels = mid_channels
    select_weights_from_index(module.conv2, selected_index, 'conv', dim=1)

    print(f"new layer: {module}")
    return True


def do_selfatt_prune(module, device, keep_heads_ratio=0.5, min_heads=1):
    inner_dim = module.inner_dim
    heads = module.heads
    dim_head = inner_dim // heads
    keep_heads = int(heads * keep_heads_ratio)
    if keep_heads <= min_heads:
        print("meet minimum heads, not prune this seflatt block")
        return False

    print(f"origin number of heads={heads}, dim_head={dim_head}, new number of heads={keep_heads}")
    module.heads = keep_heads
    module.inner_dim = dim_head * keep_heads

    # TODO: calculate importance score based on to_q.weight
    selected_index = torch.range(0, int(keep_heads * dim_head)-1, dtype=torch.int64).to(device)
    select_weights_from_index(module.to_q, selected_index)
    select_weights_from_index(module.to_k, selected_index)
    select_weights_from_index(module.to_v, selected_index)
    select_weights_from_index(module.to_out[0], selected_index, dim=1)

    print(f"new layer: {module}")
    return True

def do_crossatt_prune(module, device, keep_heads_ratio=0.5, min_heads=1):
    print(f"old layer: {module}")

    inner_dim = module.inner_dim
    heads = module.heads
    keep_heads = int(heads * keep_heads_ratio)

    if keep_heads <= min_heads:
        print("meet minimum heads, not prune this crossatt block")
        return False

    dim_head = inner_dim // heads
    print(f"origin number of heads={heads}, dim_head={dim_head}, new number of heads={keep_heads}")
    module.heads = keep_heads
    module.inner_dim = dim_head * keep_heads

    selected_index = torch.range(0, int(keep_heads * dim_head)-1, dtype=torch.int64).to(device)
    select_weights_from_index(module.to_q, selected_index)
    select_weights_from_index(module.to_k, selected_index)
    select_weights_from_index(module.to_v, selected_index)
    select_weights_from_index(module.to_out[0], selected_index, dim=1)

    print(f"new layer: {module}")
    return True


def do_mlp_prune(module, device, keep_ratio=0.5):
    p = 2
    print(f"origin layer: {module}")
    out_features = module.net[0].proj.out_features
    new_out_features = int(out_features * keep_ratio)
    
    selected_index = torch.range(0, new_out_features-1, dtype=torch.int64).to(device)
    print("selected_index[-1]=", selected_index[-1])

    select_weights_from_index(module.net[0].proj, selected_index, dim=0)

    selected_index = torch.range(0, new_out_features // 2 - 1, dtype=torch.int64).to(device)
    print("selected_index[-1]=", selected_index[-1])
    select_weights_from_index(module.net[2], selected_index, dim=1)

    print(f"new layer: {module}")

def check_valid(layer_ids, start, end):
    return any(x in list(range(start, end+1)) for x in layer_ids)

def get_dataloader(dataset, pipeline, batch_size=8):
    # Get the true column names for input/target.
    column_names = dataset.column_names
    dataset_columns = ("image", "text")
    image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    tokenizer = pipeline.pipe.tokenizer
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    # Preprocessing the datasets.
    from torchvision import transforms
    train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}
        
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=0,
    )
    return train_dataloader
    
def calculate_score(latents):
    def get_latent_avg_std(latent):
        avg = np.mean(latent, axis=0)
        std = np.std(latent, axis=0)
        return avg, std

    org_latent = latents[0]
    org_avg, org_std = get_latent_avg_std(org_latent)
    scores = []
    for mod_latent in latents[1:]:
        mod_avg, mod_std = get_latent_avg_std(mod_latent)
        dis_avg = np.linalg.norm(org_avg - mod_avg)
        dis_std = np.linalg.norm(org_std - mod_std)
        score = dis_avg + dis_std
        scores.append(score)
    return scores

def prune_unet(args, unet):
    origin_modules = {}
    print('='*20 + ' prune unet ' + '='*20)
    success = False
    if args.prune_resnet_layers is not None:
        if not isinstance(args.prune_resnet_layers, list):
            prune_resnet_layers = list(map(int, args.prune_resnet_layers.split(',')))
        else:
            prune_resnet_layers = args.prune_resnet_layers
        if not check_valid(prune_resnet_layers, 1, 12):
            raise ValueError(f"prune_resnet_layers = {prune_resnet_layers} is invalid. Values should between [1, 12]")
        n_resnet_block = 0
        for name, module in unet.named_modules():
            if isinstance(module, ResnetBlock2D):
                n_resnet_block += 1
                if n_resnet_block not in prune_resnet_layers:
                    continue
                origin_modules[name] = deepcopy(module)
                success = do_resnet_prune(module, unet.device, keep_ratio=args.keep_resnet_ratio)

        print(f"total blocks = {n_resnet_block}, prune_resnet_layers ids = {prune_resnet_layers}")

    if args.prune_selfattn_layers is not None:
        if not isinstance(args.prune_selfattn_layers, list):
            prune_selfattn_layers = list(map(int, args.prune_selfattn_layers.split(',')))
        else:
            prune_selfattn_layers = args.prune_selfattn_layers
        if not check_valid(prune_selfattn_layers, 1, 9):
            raise ValueError(f"prune_selfattn_layers = {prune_selfattn_layers} is invalid. Values should between [1, 9]")
        n_selfattn_block = 0
        for name, module in unet.named_modules():
            if isinstance(module, BasicTransformerBlock):
                n_selfattn_block += 1
                if n_selfattn_block not in prune_selfattn_layers:
                    continue
                origin_modules[name] = deepcopy(module)
                success = do_selfatt_prune(module.attn1, unet.device, keep_heads_ratio=args.keep_selfattn_heads_ratio)    

        print(f'total selfatt blocks = {n_selfattn_block}, prune_selfattn_layers ids = {prune_selfattn_layers}')

    if args.prune_crossattn_layers is not None:
        if not isinstance(args.prune_crossattn_layers, list):
            prune_crossattn_layers = list(map(int, args.prune_crossattn_layers.split(',')))
        else:
            prune_crossattn_layers = args.prune_crossattn_layers
        if not check_valid(prune_crossattn_layers, 1, 9):
            raise ValueError(f"prune_crossattn_layers = {prune_crossattn_layers} is invalid. Values should between [1, 9]")
        n_crossattn_block = 0
        for name, module in unet.named_modules():
            if isinstance(module, BasicTransformerBlock):
                n_crossattn_block += 1
                if n_crossattn_block not in prune_crossattn_layers:
                    continue
                origin_modules[name] = deepcopy(module)
                success = do_crossatt_prune(module.attn2, unet.device, keep_heads_ratio=args.keep_crossattn_heads_ratio)    

        print(f'total corssatt blocks = {n_crossattn_block}, prune_corssatt_layers ids = {prune_crossattn_layers}')
        
    if args.prune_mlp_layers is not None:
        n_mlp_layers = 0
        prune_mlp_layers = list(map(int, args.prune_mlp_layers.split(',')))
        for name, module in unet.named_modules():
            if isinstance(module, FeedForward):
                n_mlp_layers += 1
                if n_mlp_layers not in prune_mlp_layers:
                    continue
                success = do_mlp_prune(module, unet.device, keep_ratio=args.keep_mlp_ratio)    

        print(f'total mlp layers = {n_mlp_layers}, prune_mlp_layers ids = {prune_mlp_layers}')

    return origin_modules, success

def recover_unet(unet, origin_modules, verbose=False, logger=None):
    print('='*20 + ' recover uent ' + '='*20)
    for name, origin_module in origin_modules.items():
        split_name = name.split('.')
        parent = split_name[:-1]
        k = split_name[-1]
        if verbose:
            print(f'before recover {name}', unet.get_submodule(name))
        unet.get_submodule(('.').join(parent))[int(k)] = origin_module
        if verbose:
            print(f'after recover {name}', unet.get_submodule(name))

def set_choice(args, choice):
    if choice[0] == 'resnet':
        args.prune_resnet_layers = [choice[1]]
    if choice[0] == 'crossattn':
        args.prune_crossattn_layers = [choice[1]]
    if choice[0] == 'selfattn':
        args.prune_selfattn_layers = [choice[1]]
    