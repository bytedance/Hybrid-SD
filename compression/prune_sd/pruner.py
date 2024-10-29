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
import typing
from diffusers.models.attention import FeedForward
from copy import deepcopy
from .prune_utils import get_select_index, padding_to_multiple, select_weights_from_index
from .calflops import calculate_flops
import os
import random
from diffusers.models.modeling_utils import load_state_dict
from copy import deepcopy

from .models.resnet import ResnetBlock2D
from .models.transformer_2d import BasicTransformerBlock

import logging

logger = logging.getLogger(__name__)

class UnetPruner:
    def __init__(self, args, unet, example_inputs, ema_unet=None, importance='l2', verbose=False):
        self.unet = unet
        self.args = args
        self.device = unet.device
        # config for bk-sdm-small
        if args.base_arch == 'bk-sdm-small':
            self.unet._internal_dict['block_mid_channels'] = [320, 640, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 320, 320] 
        elif args.base_arch == 'bk-sdm-tiny':
            self.unet._internal_dict['block_mid_channels'] = [320, 640, 1280, 1280, 1280, 640, 640, 320, 320]
        else:
            raise NotImplementedError

        self.unet._internal_dict['block_selfattn_heads'] = [8] * 9
        self.unet._internal_dict['block_crossattn_heads'] = [8] * 9
        
        self.prune_resnet_layers = None
        if isinstance(args.keep_resnet_ratio, list):
            self.keep_renset_ratio = args.keep_resnet_ratio
            print(f"keep resnet ratio = {self.keep_renset_ratio}")
        else:
            self.keep_renset_ratio = [args.keep_resnet_ratio] * len(self.unet._internal_dict['block_mid_channels'])
        self.prune_crossattn_layers = None
        if isinstance(args.keep_crossattn_heads_ratio, list):
            self.keep_crossattn_heads_ratio = args.keep_crossattn_heads_ratio
            print(f"keep_crossattn_heads_ratio = {self.keep_crossattn_heads_ratio}")
        else:
            self.keep_crossattn_heads_ratio = [args.keep_crossattn_heads_ratio] * len(self.unet._internal_dict['block_crossattn_heads'])
        self.prune_selfattn_layers = None
        if isinstance(args.keep_selfattn_heads_ratio, list):
            self.keep_selfattn_heads_ratio = args.keep_selfattn_heads_ratio
            print(f"keep_selfattn_heads_ratio = {self.keep_selfattn_heads_ratio}")
        else:
            self.keep_selfattn_heads_ratio = [args.keep_selfattn_heads_ratio] * len(self.unet._internal_dict['block_selfattn_heads'])
        self.example_inputs = example_inputs
        
        self.candidate_choices = [('resnet', i) for i in range(1, 13)] + [('crossattn', i) for i in range(1, 10)] + [('selfattn', i) for i in range(1, 10)]
        # self.candidate_choices = [('resnet', i) for i in range(1, 4)]
        self.origin_modules = dict()
        self.ema_unet = ema_unet
        self.verbose = verbose

        self.init()
        
    def init(self):
        do_prune = False
        if self.args.prune_resnet_layers is not None:
            do_prune = True
            self.prune_resnet_layers = list(map(int, self.args.prune_resnet_layers.split(',')))
            print(f"prune_resnet_layers={self.prune_resnet_layers}")
        if self.args.prune_selfattn_layers is not None:
            do_prune = True
            self.prune_selfattn_layers = list(map(int, self.args.prune_selfattn_layers.split(',')))
            print(f"prune_selfattn_layers={self.prune_selfattn_layers}")
        if self.args.prune_crossattn_layers is not None:
            do_prune = True
            self.prune_crossattn_layers = list(map(int, self.args.prune_crossattn_layers.split(',')))
            print(f"prune_crossattn_layers={self.prune_crossattn_layers}")
        print(f'do_prune = {do_prune}')

        if do_prune == True:
            self.prune(change_config=True)

    def set_choice(self, choice):
        if choice[0] == 'resnet':
            self.prune_resnet_layers = [choice[1]]
        if choice[0] == 'crossattn':
            self.prune_crossattn_layers = [choice[1]]
        if choice[0] == 'selfattn':
            self.prune_selfattn_layers = [choice[1]]
        
    def reset_choice(self):
        self.prune_resnet_layers = None
        self.prune_crossattn_layers = None
        self.prune_selfattn_layers = None

    def recover(self):
        if self.origin_modules is not None:
            for name, origin_module in self.origin_modules.items():
                split_name = name.split('.')
                parent = split_name[:-1]
                k = split_name[-1]
                self.unet.get_submodule(('.').join(parent))[int(k)] = origin_module
        
        self.origin_modules = dict()
        self.reset_choice()

    def prune_resnet(self, module, keep_resnet_ratio, min_channels=64):
        num_groups = 32

        # print(f"origin layer: {module}")
        # calculate importance score based on conv1.weight
        out_channels = module.conv1.out_channels
        mid_channels = padding_to_multiple(int(out_channels * keep_resnet_ratio), num_groups)
        if mid_channels <= min_channels:
            # logger.info("meet minimum channels, not prune this resnet block")
            return False

        n_pruned_channels = out_channels - mid_channels
        print(f"Prune channel {out_channels} -> {mid_channels}")
        keep_idxs = get_select_index(module.conv1.weight, out_channels, n_pruned_channels, self.device)
        sub_module_names = ['conv1', 'time_emb_proj', 'norm2']
        sub_module_types = ['conv', 'fc', 'norm']
        for sub_module_name, sub_module_type in zip(sub_module_names, sub_module_types):
            sub_module = getattr(module, sub_module_name)
            select_weights_from_index(sub_module, keep_idxs, sub_module_type)

        module.conv2.in_channels = mid_channels
        select_weights_from_index(module.conv2, keep_idxs, 'conv', dim=1)
        # print(f"new layer: {module}")
        return True

    def prune_attention(self, module, keep_heads_ratio, min_heads=2):
        inner_dim = module.inner_dim
        heads = module.heads
        dim_head = inner_dim // heads
        keep_heads = int(heads * keep_heads_ratio)
        if keep_heads <= min_heads:
            # logger.info("meet minimum heads, not prune this seflatt block")
            return False

        # print(f"origin number of heads={heads}, dim_head={dim_head}, new number of heads={keep_heads}")
        print(f"Prune attention heads {heads} -> {keep_heads}")
        module.heads = keep_heads
        module.inner_dim = dim_head * keep_heads

        # TODO: calculate importance score based on to_q.weight
        keep_idxs = torch.range(0, int(keep_heads * dim_head)-1, dtype=torch.int64).to(self.device)
        for sub_module_name in ['to_q', 'to_k', 'to_v']:
            sub_module = getattr(module, sub_module_name)
            select_weights_from_index(sub_module, keep_idxs)
        select_weights_from_index(module.to_out[0], keep_idxs, dim=1)
        return True
        
        
    def prune_selfattn(self, module, keep_ratio):
        success = self.prune_attention(module, keep_ratio)
        return success


    def prune_crossattn(self, module, keep_ratio):
        success = self.prune_attention(module, keep_ratio)
        return success
        
    def prune(self, change_config=False):
        # print('='*20 + ' prune unet ' + '='*20)
        success = False
        if self.prune_resnet_layers is not None:
            n_resnet_block = 0
            for name, module in self.unet.named_modules():
                if isinstance(module, ResnetBlock2D):
                    n_resnet_block += 1
                    if n_resnet_block not in self.prune_resnet_layers:
                        continue
                    self.origin_modules[name] = deepcopy(module)
                    success = self.prune_resnet(module, self.keep_renset_ratio[n_resnet_block - 1])
                    # logger.info(f'Prune resnet {name} {success}')

                    if change_config:
                        # logger.info(f"prune {name} {self.origin_modules[name].conv1.out_channels} -> {module.conv1.out_channels}")
                        self.unet._internal_dict['block_mid_channels'][n_resnet_block-1] = module.conv1.out_channels

        if self.prune_selfattn_layers is not None:
            n_selfattn_block = 0
            for name, module in self.unet.named_modules():
                if isinstance(module, BasicTransformerBlock):
                    n_selfattn_block += 1
                    if n_selfattn_block not in self.prune_selfattn_layers:
                        continue
                    self.origin_modules[name] = deepcopy(module)
                    success = self.prune_selfattn(module.attn1, self.keep_selfattn_heads_ratio[n_selfattn_block - 1])
                    # logger.info(f'Prune self attention {name} {success}')
                    if change_config:
                        # logger.info(f"prune {name} {self.origin_modules[name].attn1.heads} -> {module.attn1.heads}")
                        self.unet._internal_dict['block_selfattn_heads'][n_selfattn_block-1] = module.attn1.heads

            # print(f'total selfatt blocks = {n_selfattn_block}, prune_selfattn_layers ids = {self.prune_selfattn_layers}')

        if self.prune_crossattn_layers is not None:
            n_crossattn_block = 0
            for name, module in self.unet.named_modules():
                if isinstance(module, BasicTransformerBlock):
                    n_crossattn_block += 1
                    if n_crossattn_block not in self.prune_crossattn_layers:
                        continue
                    self.origin_modules[name] = deepcopy(module)
                    success = self.prune_crossattn(module.attn2, self.keep_crossattn_heads_ratio[n_crossattn_block - 1])
                    # logger.info(f'Prune cross attention {name} {success}')
                    if change_config:
                        # logger.info(f"prune {name} {self.origin_modules[name].attn2.heads} -> {module.attn2.heads}")
                        self.unet._internal_dict['block_crossattn_heads'][n_crossattn_block-1] = module.attn2.heads

            # print(f'total corssatt blocks = {n_crossattn_block}, prune_corssatt_layers ids = {self.prune_crossattn_layers}')

        return success


    def sample(self):
        choice = random.choice(self.candidate_choices)
        return choice

        
    def calc_flops(self, output_as_string=False):
        flops, macs, params = calculate_flops(
            model=self.unet, 
            args=self.example_inputs,
            output_as_string=output_as_string,
            output_precision=4,
            print_detailed=False,
            print_results=False)
        return flops, macs, params