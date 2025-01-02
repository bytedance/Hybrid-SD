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

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from torchvision.transforms import  GaussianBlur
import torch.nn.functional as F
from compression.optimize_vae.models.litevae.wavelet import DWTH
import random
import pdb
import functools


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)





class NLayerDiscriminator_condtion(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=7, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator_condtion, self).__init__()
        
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        # self.process_cond_layer = nn.Conv2d(4, 3, 3, stride=1, padding=1)
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)




    def forward(self, input, cond):
        """Standard forward."""
        # cond = self.process_cond_layer(cond)
        cond = cond.repeat(1, 1, input.shape[-1]//cond.shape[-1], input.shape[-1]//cond.shape[-1]) #(1,4,64,64)
        input = torch.cat((input,cond), dim=1)
        return self.main(input)
        """[Conv2d(7, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), LeakyReLU(negative_slope=0.2, inplace=True), 
        Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False), LeakyReLU(negative_slope=0.2, inplace=True),
        Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False), LeakyReLU(negative_slope=0.2, inplace=True),
        Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)), InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False), LeakyReLU(negative_slope=0.2, inplace=True),
        Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))]"""




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


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = torch.ones(size=()) * logvar_init #nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    #self.decoder.conv_out.weight (last_layer:self.decoder.conv_out.weight)
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

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            log = {"total_loss": loss.clone().detach().mean(), "logvar": self.logvar.detach(),
                   "kl_loss": kl_loss.detach().mean(), "nll_loss": nll_loss.detach().mean(),
                   "rec_loss": rec_loss.detach().mean(),
                   "d_weight": d_weight.detach(),
                   "{disc_factor": torch.tensor(disc_factor),
                   "g_loss": g_loss.detach().mean(),
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

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log

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



############# use only gan after reconstruction
class LPIPSWithDiscriminator_decoder_only_gan(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 if_use_average=False,if_penalize=0,if_use_highfrec=False,
                 if_maxpool = False,bianyuanmodel = None,gan_weight=1.0, penalize_weight = 1.0,replay_buffer = False):
                

        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = torch.ones(size=()) * logvar_init #nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        # self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.if_use_average = if_use_average
        self.p_loss_init = torch.tensor(0.0)
        self.high_frec_init = torch.tensor(0.0)
        self.if_penalize = if_penalize
        self.mseloss = nn.MSELoss()
        self.wavelet = DWTH(J=1, mode='zero', wave='haar')  
        self.charb = CharbonnierLoss(out_norm="")
        self.if_use_highfrec = if_use_highfrec
        self.if_maxpool = if_maxpool
        self.maxpool_loss  = torch.tensor(0.0)
        self.rec_loss  = torch.tensor(0.0)
        self.gloss_init = torch.tensor(0.0)
        self.bianyuanmodel = bianyuanmodel
        self.mean_test = torch.tensor([104.007, 116.669, 122.679])
        self.bianyuanloss_init = torch.tensor(0.0)
        self.gan_weight = gan_weight
        self.penalize_weight = penalize_weight
        self.replay_buffer = replay_buffer
        self.buff = []
        self.max_buff_len = 10000
        
    def disc_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(1. - logits_real) 
        loss_fake = torch.mean(1. + logits_fake) 
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def high_frec_loss(self,inputs, reconstructions):
        wave_input = self.wavelet(inputs.contiguous()) 
        wave_rec = self.wavelet(reconstructions.contiguous()) 
        wavelet_loss = self.charb(wave_input, wave_rec)
        return wavelet_loss
    

    def output_bianyuan_img(self,inputs):
        if len(inputs.shape) < 4:
            inputs = inputs.unsqueeze(0)


        inputs = inputs.add(1.).div(2.).mul(255)
        inputs = inputs.permute(0, 2, 3, 1)[:,:, [0,3,2,1]]
        inputs = inputs - self.mean_test.to(inputs.device)
        inputs = inputs.permute(0,3,1,2)

        inputs = self.bianyuanmodel(inputs, single_test=False) ### list
        edge_maps = []
        for i in inputs:
            tmp = torch.sigmoid(i)
            edge_maps.append(tmp)

        tensor = torch.cat(edge_maps)
        tmp = 1-tensor[tmp.shape[0]-1, 0, ...]
        return tmp
        


    #self.decoder.conv_out.weight (last_layer:self.decoder.conv_out.weight)
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



    def forward(self, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None,logger=None):

        if global_step < self.discriminator_iter_start:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                p_loss = self.perceptual_weight * p_loss


            nll_loss= (rec_loss + p_loss) / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss

            if weights is not None:
                weighted_nll_loss = weights*nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        else:
            weighted_nll_loss = self.rec_loss

            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log





############# use only conditional gan after reconstruction
class LPIPSWithDiscriminator_decoder_only_gan_condition(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 if_use_average=False,if_penalize=0,if_use_highfrec=False,
                 if_maxpool = False,bianyuanmodel = None,gan_weight=1.0, penalize_weight = 1.0,replay_buffer = False):
                

        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = torch.ones(size=()) * logvar_init #nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator_condtion()
        self.discriminator_iter_start = disc_start
        # self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.if_use_average = if_use_average
        self.p_loss_init = torch.tensor(0.0)
        self.high_frec_init = torch.tensor(0.0)
        self.if_penalize = if_penalize
        self.mseloss = nn.MSELoss()
        self.wavelet = DWTH(J=1, mode='zero', wave='haar')  
        self.charb = CharbonnierLoss(out_norm="")
        self.if_use_highfrec = if_use_highfrec
        self.if_maxpool = if_maxpool
        self.maxpool_loss  = torch.tensor(0.0)
        self.rec_loss  = torch.tensor(0.0)
        self.gloss_init = torch.tensor(0.0)
        self.bianyuanmodel = bianyuanmodel
        self.mean_test = torch.tensor([104.007, 116.669, 122.679])
        self.bianyuanloss_init = torch.tensor(0.0)
        self.gan_weight = gan_weight
        self.penalize_weight = penalize_weight
        self.replay_buffer = replay_buffer
        self.buff = []
        self.buff_cond = []
        self.max_buff_len = 10000

        
    def disc_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(1. - logits_real) 
        loss_fake = torch.mean(1. + logits_fake) #
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def high_frec_loss(self,inputs, reconstructions):
        wave_input = self.wavelet(inputs.contiguous()) 
        wave_rec = self.wavelet(reconstructions.contiguous()) 
        wavelet_loss = self.charb(wave_input, wave_rec)
        return wavelet_loss
    

    def output_bianyuan_img(self,inputs):
        if len(inputs.shape) < 4:
            inputs = inputs.unsqueeze(0)


        inputs = inputs.add(1.).div(2.).mul(255)
        inputs = inputs.permute(0, 2, 3, 1)[:,:, [0,3,2,1]]
        inputs = inputs - self.mean_test.to(inputs.device)
        inputs = inputs.permute(0,3,1,2)

        inputs = self.bianyuanmodel(inputs, single_test=False) ### list
        edge_maps = []
        for i in inputs:
            tmp = torch.sigmoid(i)
            edge_maps.append(tmp)

        tensor = torch.cat(edge_maps)
        tmp = 1-tensor[tmp.shape[0]-1, 0, ...]
        return tmp
        


    #self.decoder.conv_out.weight (last_layer:self.decoder.conv_out.weight)
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



    def forward(self, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None,logger=None):

        if global_step < self.discriminator_iter_start:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
                p_loss = self.perceptual_weight * p_loss


            nll_loss= (rec_loss + p_loss) / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss

            if weights is not None:
                weighted_nll_loss = weights*nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        else:
            weighted_nll_loss = self.rec_loss


        # now the GAN part
        if optimizer_idx == 0:
            # generator update

            if global_step < self.discriminator_iter_start:
                g_loss = self.gloss_init
                penalize_loss = self.gloss_init
            else:
                
                logits_fake = self.discriminator(reconstructions.contiguous(), cond.contiguous())
                g_loss = -torch.mean(logits_fake) 

                ##### 
                if self.if_penalize>0:
                    logits_real = self.discriminator(inputs.contiguous(), cond.contiguous())
                    penalize_loss = self.mseloss(logits_fake,logits_real)
                    # penalize_loss = torch.sum(torch.abs(logits_fake.contiguous() - logits_real.contiguous())) 
                    # penalize_loss = penalize_loss / penalize_loss.shape[0]
                else:
                    penalize_loss = torch.tensor(0.0)


            g_loss =  g_loss *  self.gan_weight
            penalize_loss =  penalize_loss * self.penalize_weight
            loss = weighted_nll_loss  + g_loss + penalize_loss 
            log = {"total_loss": loss.clone().detach().mean(), "logvar": self.logvar.detach(),
                   "weighted_nll_loss":weighted_nll_loss.detach().mean(),
                   "penalize_loss":penalize_loss.detach().mean(),
                   "g_loss": g_loss.detach().mean()
                   }
            return loss, log


        if optimizer_idx == 1:
            # second pass for discriminator update
            # add new fake / ctx to buffer
            if self.replay_buffer:
                inputs = inputs.contiguous().detach()
                reconstructions = reconstructions.contiguous().detach()
                cond = cond.contiguous().detach()
                # logger.info(f'############ before reconstructions shape = {reconstructions.shape}')
                # logger.info(f'############ before cond shape = {cond.shape}')

                for fake_i, ctx_i in zip(reconstructions, cond):
                    if len(self.buff) >= self.max_buff_len:
                        i = random.randrange(0, len(self.buff))
                        self.buff[i][0].copy_(fake_i.clone().to(torch.device("cpu")))
                        self.buff[i][1].copy_(ctx_i.clone().to(torch.device("cpu")))
                    else:
                        self.buff.append((fake_i.clone().to(torch.device("cpu")), ctx_i.clone().to(torch.device("cpu"))))

                # sample half of fake / ctx new, half from buffer
                n = len(reconstructions) // 2
                fake_shuf, fake_shuf_ctx = (
                    torch.stack(items, 0)
                    for items in zip(*(random.choice(self.buff) for _ in range(n)))
                )
                reconstructions = torch.cat([reconstructions[:n], fake_shuf.to(reconstructions.device)], 0)
                cond = torch.cat([cond[:n], fake_shuf_ctx.to(cond.device)], 0)

            # logger.info(f'############ reconstructions shape = {reconstructions.shape}')
            # logger.info(f'############ cond shape = {cond.shape}')
            logits_real = self.discriminator(inputs.contiguous(), cond.contiguous())
            logits_fake = self.discriminator(reconstructions.contiguous(), cond.contiguous())
            d_loss = self.disc_loss(logits_real, logits_fake)
            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log





class LPIPSWithDiscriminator_decoder_only_finetune(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = torch.ones(size=()) * logvar_init #nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    #self.decoder.conv_out.weight (last_layer:self.decoder.conv_out.weight)
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

    def forward(self, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss  + d_weight * disc_factor * g_loss
            log = {"total_loss": loss.clone().detach().mean(), "logvar": self.logvar.detach(),
                   "nll_loss": nll_loss.detach().mean(),
                   "rec_loss": rec_loss.detach().mean(),
                   "d_weight": d_weight.detach(),
                   "{disc_factor": torch.tensor(disc_factor),
                   "g_loss": g_loss.detach().mean(),
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

            #disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            disc_factor = self.disc_factor
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log

