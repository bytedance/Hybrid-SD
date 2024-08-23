import torch
import torch.nn as nn
import random

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from torchvision.transforms import  GaussianBlur
import torch.nn.functional as F 
from compression.optimize_vae.models.litevae.wavelet import  DWTH
from torch.nn.utils import spectral_norm
from publics.stylegant.networks.discriminator import VAEDiscrminator




###############################
## unet discriminator
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


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


##
## remove Lpips loss, use Huber loss

def Pseudo_Huber_loss(x, y):
    c = torch.tensor(0.03)
    return torch.sum( (torch.sqrt((x - y)**2 + c**2) -c), dim=(1,2,3))

class LitevaeWithDino(nn.Module):
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

        self.discriminator = VAEDiscrminator(c_dim=0)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.gaussian =   GaussianBlur((7,7), sigma=(0.1, 2.0))
        self.wavelet =  DWTH(J=1, mode='zero', wave='haar')  
        self.charb = CharbonnierLoss(out_norm="")

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

    def high_frec_loss(self, inputs, reconstructions):
        high_input = inputs - self.gaussian(inputs)
        high_rec = reconstructions - self.gaussian(reconstructions)
        gaussian_loss = F.l1_loss(high_input, high_rec, reduction="sum")
        wave_input = self.wavelet(inputs)
        wave_rec = self.wavelet(reconstructions)
        wavelet_loss = self.charb(wave_input, wave_rec)
        return gaussian_loss + wavelet_loss
    
 

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, cond=None, split="train",
                weights=None):
        B,C,H,W = inputs.shape
        
        #  Pseudo_Huber_loss
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
            g_loss =  -torch.mean(logits_fake)*(B*C*H*W)

                

            high_fre_loss = self.high_frec_loss(inputs, reconstructions)
            loss = weighted_nll_loss + self.kl_weight * kl_loss +  0.05 * g_loss + 0.1 * high_fre_loss
            log = {"total_loss":loss.clone().detach().mean(),
                   "kl_loss": kl_loss.detach().mean(), "nll_loss":nll_loss.detach().mean(),
                   "rec_loss": rec_loss.detach().mean(),
                   "g_loss": g_loss.detach().mean(),
                   "high_fre_loss":high_fre_loss.detach().mean()
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
        self.gaussian =   GaussianBlur((7,7), sigma=(0.1, 2.0))
        self.wavelet =  DWTH(J=1, mode='zero', wave='haar')  
        self.charb = CharbonnierLoss(out_norm="bci")
        #self.rec = LPIPS().eval()
        self.rec = torch.nn.MSELoss()
        #self.rec = CharbonnierLoss(out_norm="bci")
        #self.rec = torch.nn.L1Loss(reduction="mean")

    # def high_frec_loss(self, inputs, reconstructions):
    #     high_input = inputs - self.gaussian(inputs)
    #     high_rec = reconstructions - self.gaussian(reconstructions)
    #     gaussian_loss = F.l1_loss(high_input, high_rec, reduction="mean")
    #     wave_input = self.wavelet(inputs)
    #     wave_rec = self.wavelet(reconstructions)
    #     wavelet_loss = self.charb(wave_input, wave_rec)
    #     return gaussian_loss + wavelet_loss
    
    def high_frec_loss(self, inputs, reconstructions):
        wave_input = self.wavelet(inputs)
        wave_rec = self.wavelet(reconstructions)
        wavelet_loss = self.charb(wave_input, wave_rec)
        return wavelet_loss
 
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
      
        rec_loss =  self.rec(F.adaptive_avg_pool2d(inputs, 16), F.adaptive_avg_pool2d(reconstructions, 16) )
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(reconstructions.contiguous())
            logits_real = self.discriminator(inputs.contiguous().detach())

            
            g_loss = torch.abs(logits_real.mean() - logits_fake.mean())
            g_weight = 1.
            if global_step < self.discriminator_iter_start:
                g_weight = 0.0
            else:
                g_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer)

            #high_fre_loss = self.high_frec_loss(inputs, reconstructions)
            loss = rec_loss  +  0.005 * g_weight * g_loss #+ 0.02 * high_fre_loss
            log = {"total_loss":loss.clone().detach().mean(),
                   "rec_loss": rec_loss.detach().mean(),
                   "g_loss": (g_loss*0.005*g_weight).detach().mean(),
                   #"high_fre_loss":high_fre_loss.detach().mean()*0.02
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            d_loss =  self.disc_loss(logits_real, logits_fake)

            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log


## version 2 mseloss + gloss
class TinyVaeDino2(nn.Module):
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
        self.gaussian =   GaussianBlur((7,7), sigma=(0.1, 2.0))
        self.wavelet =  DWTH(J=1, mode='zero', wave='haar')  
        self.charb = CharbonnierLoss(out_norm="bci")
        #self.rec = LPIPS().eval()
        self.rec = torch.nn.MSELoss()
        #self.rec = CharbonnierLoss(out_norm="bci")
        #self.rec = torch.nn.L1Loss(reduction="mean")

    # def high_frec_loss(self, inputs, reconstructions):
    #     high_input = inputs - self.gaussian(inputs)
    #     high_rec = reconstructions - self.gaussian(reconstructions)
    #     gaussian_loss = F.l1_loss(high_input, high_rec, reduction="mean")
    #     wave_input = self.wavelet(inputs)
    #     wave_rec = self.wavelet(reconstructions)
    #     wavelet_loss = self.charb(wave_input, wave_rec)
    #     return gaussian_loss + wavelet_loss
    
    def high_frec_loss(self, inputs, reconstructions):
        wave_input = self.wavelet(inputs)
        wave_rec = self.wavelet(reconstructions)
        wavelet_loss = self.charb(wave_input, wave_rec)
        return wavelet_loss
 
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
      
        rec_loss =  self.rec(F.adaptive_avg_pool2d(inputs, 8), F.adaptive_avg_pool2d(reconstructions, 8) )
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(reconstructions.contiguous())
            logits_real = self.discriminator(inputs.contiguous().detach())
            
            #g_loss = - logits_fake.mean()
            g_loss = torch.abs((logits_real.detach() - logits_fake)).mean()
            g_weight = 1.
            if global_step < self.discriminator_iter_start:
                g_weight = 0.0
            # else:
            #     g_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer)

            #high_fre_loss = self.high_frec_loss(inputs, reconstructions)
            #loss = rec_loss  +  0.05 * g_weight * g_loss #+ 0.02 * high_fre_loss
            loss = rec_loss + g_loss 
            log = {"total_loss":loss.clone().detach().mean(),
                   "rec_loss": rec_loss.detach().mean(),
                   "g_loss": (g_loss*0.5*g_weight).detach().mean(),
                   #"high_fre_loss":high_fre_loss.detach().mean()*0.02
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            d_loss =  self.disc_loss(logits_real, logits_fake)
            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log


## version 3 with conditions perceptual loss
class TinyVaeDino3(nn.Module):
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
        self.gaussian =   GaussianBlur((7,7), sigma=(0.1, 2.0))
        self.wavelet =  DWTH(J=1, mode='zero', wave='haar').cuda()  
        self.charb = CharbonnierLoss(out_norm="bci")
        self.rec = LPIPS().eval()
        #self.rec = torch.nn.MSELoss()
        #self.rec = CharbonnierLoss(out_norm="bci")
        #self.rec = torch.nn.L1Loss(reduction="mean")
        #self.high_d = torch.nn.L1Loss(reduction="mean")

    # def high_frec_loss(self, inputs, reconstructions):
    #     high_input = inputs - self.gaussian(inputs)
    #     high_rec = reconstructions - self.gaussian(reconstructions)
    #     gaussian_loss = F.l1_loss(high_input, high_rec, reduction="mean")
    #     wave_input = self.wavelet(inputs)
    #     wave_rec = self.wavelet(reconstructions)
    #     wavelet_loss = self.charb(wave_input, wave_rec)
    #     return gaussian_loss + wavelet_loss
    
    # def high_frec_loss(self, inputs, reconstructions):
    #     wave_input = self.wavelet(inputs)
    #     wave_rec = self.wavelet(reconstructions)
    #     #wavelet_loss = self.charb(wave_input, wave_rec)
    #     wavelet_loss = self.high_d(wave_input, wave_rec)
    #     return wavelet_loss
 
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
            logits_fake = self.discriminator(torch.cat([reconstructions.contiguous(), inputs.contiguous().detach()],dim=0))
            #logits_real = self.discriminator(torch.cat([inputs.contiguous().detach(), inputs.contiguous().detach()], dim=0))
            
            #g_loss = - logits_fake.mean()
            #g_loss = torch.abs((logits_real.detach() - logits_fake)).mean()
            g_loss = - torch.mean(logits_fake)
            g_weight = 1.
            if global_step < self.discriminator_iter_start:
                g_weight = 0.0
            else:
                g_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer)

            #high_fre_loss = self.high_frec_loss(inputs, reconstructions)
            #loss = rec_loss  +  0.05 * g_weight * g_loss #+ 0.02 * high_fre_loss
            loss = rec_loss + 0.05* g_weight* g_loss 
            log = {"total_loss":loss.clone().detach().mean(),
                   "rec_loss": rec_loss.detach().mean(),
                   "g_loss": (g_loss*0.1*g_weight).detach().mean(),
                   #"high_fre_loss":high_fre_loss.detach().mean()*0.02
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_fake = self.discriminator(torch.cat([reconstructions.contiguous().detach(), inputs.contiguous().detach()],dim=0))
            logits_real = self.discriminator(torch.cat([inputs.contiguous().detach(), inputs.contiguous().detach()], dim=0))
            d_loss =  self.disc_loss(logits_real, logits_fake)
            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log


## version 4 perceptual loss
class TinyVaeDino4(nn.Module):
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



## version 5 perceptual loss (dino input should be [-1,1])
class TinyVaeDino5(nn.Module):
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
            inputs, reconstructions = inputs.mul(2.).sub(1.), reconstructions.mul(2.).sub(1.)
            logits_fake = self.discriminator(torch.cat([reconstructions.contiguous(), inputs.contiguous().detach()],dim=0))
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
            inputs, reconstructions = inputs.mul(2.).sub(1.), reconstructions.mul(2.).sub(1.)
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




### version 6   v4 + replay buffer

class TinyVaeDino6(nn.Module):
    def __init__(self, disc_start,  pixelloss_weight=1.0,
                 disc_weight=1.0,
                 disc_conditional=False,
                 disc_loss="hinge", max_buff_len=16384):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.pixel_weight = pixelloss_weight

        self.discriminator = VAEDiscrminator(c_dim=0)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.rec = LPIPS().eval()
        self.buffer = []
        self.max_buff_len = max_buff_len
 
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
            logits_fake = self.discriminator(torch.cat([reconstructions.contiguous(), inputs.contiguous().detach()],dim=0))
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
            mini_batch = reconstructions.shape[0]
            rec_device = reconstructions.device
            inputs = inputs.contiguous().detach()
            reconstructions = reconstructions.contiguous().detach()
            # add to replay buffer
            for j in range(mini_batch):
                if len(self.buffer) > self.max_buff_len:
                    i = random.randrange(0, len(self.buffer))
                    self.buffer[i].copy_(reconstructions[j].clone().to(torch.device("cpu")))
                else:
                    self.buffer.append(reconstructions[j].clone().to(torch.device("cpu")))
            
            # sample half of the fake from the buffer
            n = mini_batch // 2
            
            fake_buff = torch.stack(random.sample(self.buffer, n)).to(rec_device)

            reconstructions = torch.cat([reconstructions[n:], fake_buff], 0)

            if cond is None:
                logits_real = self.discriminator(inputs)
                logits_fake = self.discriminator(reconstructions)
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
            d_loss =  self.disc_loss(logits_real, logits_fake)
            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log



## unet discriminator
class TinyVaeUNet(nn.Module):
    """
    leverage a RealESRGAN unet discriminator
    """
    def __init__(self, disc_start,  pixelloss_weight=1.0,
                 disc_weight=1.0,
                 disc_conditional=False,
                 disc_loss="vanilla"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.pixel_weight = pixelloss_weight

        self.discriminator = UNetDiscriminatorSN(num_in_ch=3, num_feat=64, skip_connection=True)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else nn.BCEWithLogitsLoss() #vanilla_d_loss

        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.gaussian =   GaussianBlur((7,7), sigma=(0.1, 2.0))
        self.wavelet =  DWTH(J=1, mode='zero', wave='haar')  
        self.charb = CharbonnierLoss(out_norm="bci")
        #self.rec = LPIPS().eval()
        self.rec = torch.nn.MSELoss()
        #self.rec = CharbonnierLoss(out_norm="bci")
        #self.rec = torch.nn.L1Loss(reduction="mean")

    # def high_frec_loss(self, inputs, reconstructions):
    #     high_input = inputs - self.gaussian(inputs)
    #     high_rec = reconstructions - self.gaussian(reconstructions)
    #     gaussian_loss = F.l1_loss(high_input, high_rec, reduction="mean")
    #     wave_input = self.wavelet(inputs)
    #     wave_rec = self.wavelet(reconstructions)
    #     wavelet_loss = self.charb(wave_input, wave_rec)
    #     return gaussian_loss + wavelet_loss
    
    # def high_frec_loss(self, inputs, reconstructions):
    #     wave_input = self.wavelet(inputs)
    #     wave_rec = self.wavelet(reconstructions)
    #     wavelet_loss = self.charb(wave_input, wave_rec)
    #     return wavelet_loss
 


    def forward(self, inputs, reconstructions,  optimizer_idx,
                global_step, cond=None, split="train", last_layer=None,
                weights=None):
      
        rec_loss =  self.rec(F.adaptive_avg_pool2d(inputs, 16), F.adaptive_avg_pool2d(reconstructions, 16))
        #rec_loss = torch.abs(inputs - reconstructions).mean()
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(reconstructions.contiguous())
            #logits_real = self.discriminator(inputs.contiguous().detach())
            
            g_loss = - logits_fake.mean()
            #g_loss = torch.abs((logits_real.detach() - logits_fake)).mean()
            #g_loss=  - torch.mean(torch.nn.functional.softplus(logits_fake))
            g_weight = 1.
            if global_step < self.discriminator_iter_start:
                g_weight = 0.0


            #high_fre_loss = self.high_frec_loss(inputs, reconstructions)
            #loss = rec_loss  +  0.05 * g_weight * g_loss #+ 0.02 * high_fre_loss
            loss = rec_loss + g_loss*0.05*g_weight 
            log = {"total_loss":loss.clone().detach().mean(),
                   "rec_loss": rec_loss.detach().mean(),
                   "g_loss": (g_loss*0.05*g_weight).detach().mean(),
                   #"high_fre_loss":high_fre_loss.detach().mean()*0.02
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            d_loss =  self.disc_loss(logits_real, logits_fake)
            log = {"disc_loss": d_loss.clone().detach().mean(),
                   "logits_real": logits_real.detach().mean(),
                   "logits_fake": logits_fake.detach().mean()
                   }
            return d_loss, log





if __name__ == "__main__":
    import pdb
    # D_net = LitevaeWithDino(disc_start=1, kl_weight=1e-5, disc_weight=0.5)
    # input  = torch.rand((2,3,512,512))
    # dec = torch.rand((2,3,512,512))
    # from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    # posteriors = DiagonalGaussianDistribution(input)
    # loss, log = D_net(input, dec, posteriors, optimizer_idx = 0, global_step=25)
    D_net = TinyVaeDino6(disc_start=1, max_buff_len=6).cuda()
    input  = torch.rand((8,3,512,512)).cuda()
    dec = torch.rand((8,3,512,512)).cuda()
    loss, log = D_net(input, dec, optimizer_idx = 1, global_step=25)
    loss, log = D_net(input, dec, optimizer_idx = 1, global_step=25)
    loss, log = D_net(input, dec, optimizer_idx = 1, global_step=25)

    print(log)

    # D_net = UNetDiscriminatorSN(num_in_ch=3).cuda()
    # input  = torch.rand((2,3,512,512)).cuda()
    # y = D_net(input)
    # print(y.shape)

    # D_net  = TinyVaeUNet(disc_start=1,disc_loss="vanilla").cuda()
    # input  = torch.rand((2,3,512,512)).cuda()
    # dec = torch.rand((2,3,512,512)).cuda()
    # loss, log  = D_net(input, dec, optimizer_idx = 0, global_step=25)
    # print(loss, log)
 
 
    # a = torch.rand(10,3,512,512)
    # b = []
    # for i in range(9):
    #     b.append(torch.rand(3,512,512))
    # b = torch.stack(random.sample(b, 5))
