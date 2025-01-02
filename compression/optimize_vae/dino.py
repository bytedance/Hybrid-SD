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
import random

from taming.modules.losses.vqperceptual import * 
from torchvision.transforms import  GaussianBlur
import torch.nn.functional as F 
from torch.nn.utils import spectral_norm
from compression.optimize_vae.models.stylegant.networks.discriminator  import VAEDiscrminator



def get_outnorm(x:torch.Tensor, out_norm:str='') -> torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if 'b' in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if 'c' in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[-3]
    if 'i' in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1]*img_shape[-2]

    return norm
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6, out_norm:str='bci'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm
    def forward(self, x, y):
        norm = get_outnorm(x, self.out_norm)
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss*norm



def Pseudo_Huber_loss(x, y):
    c = torch.tensor(0.03)
    return torch.sum( (torch.sqrt((x - y)**2 + c**2) -c), dim=(1,2,3))



def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss




class TinyVaeDino(nn.Module):
    def __init__(self, disc_start,  pixelloss_weight=1.0,
                 disc_weight=1.0,
                 disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.pixel_weight = pixelloss_weight

        self.discriminator = VAEDiscrminator(c_dim=0)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.rec = LPIPS().eval()
 
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions,  optimizer_idx,
                global_step, cond=None, split="train", last_layer=None,
                weights=None):
      

        rec_loss =  self.rec(inputs.detach(), reconstructions).mean()
        #high_loss = self.high_frec_loss(inputs, reconstructions)
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(torch.cat([reconstructions.contiguous()],dim=0))
            g_loss = - torch.mean(logits_fake)
            g_weight = 1.
            if global_step < self.discriminator_iter_start:
                g_weight = 0.0
            else:
                g_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer)


            loss = rec_loss + 0.5* g_weight* g_loss 
            log = {"total_loss":loss.clone().detach().mean(),
                   "lpips_loss": rec_loss.detach().mean(),
                   "gan_loss": (g_loss*0.5*g_weight).detach().mean(),
                   #"high_fre_loss":high_fre_loss.detach().mean()*0.02
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
            d_loss =  self.disc_loss(logits_real, logits_fake)
            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log

