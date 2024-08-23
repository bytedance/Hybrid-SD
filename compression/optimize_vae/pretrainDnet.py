import torch
import torch.nn as nn
import random

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from torchvision.transforms import  GaussianBlur
import torch.nn.functional as F 
from compression.optimize_vae.models.litevae.wavelet import  DWTH
from torch.nn.utils import spectral_norm
from publics.stylegant.networks.discriminator import VAEDiscrminator
from compression.optimize_vae.models.autoencoder_tiny import *



###################################
###  pretrianed_dnet structures

class EncoderTinyDisc(nn.Module):
    r"""
    TinyVAE without scaling
    [0,1] input [0,1] output
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: Tuple[int, ...],
        block_out_channels: Tuple[int, ...],
        act_fn: str,
    ):
        super().__init__()
        layers = []
        for i, num_block in enumerate(num_blocks):
            num_channels = block_out_channels[i]
            if i == 0:
                layers.append(nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1))
            else:
                layers.append(
                    nn.Conv2d(
                        num_channels,
                        num_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        bias=False,
                    )
                )

            for _ in range(num_block):
                layers.append(AutoencoderTinyBlock(num_channels, num_channels, act_fn))

        layers.append(nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1))

        self.layers = nn.Sequential(*layers)
        self.skip_nums = [2,6,10,13]
        self.resize_nums = [2,6,10]
        self.gradient_checkpointing = False


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `EncoderTiny` class."""
        self.mid_feats = []
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x, use_reentrant=False)
            else:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x)

        else:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i == 0:
                    resize_x = x.clone()
                if i in self.skip_nums:
                    self.mid_feats.append(x)
                    if i in self.resize_nums:
                        resize_x = F.interpolate(resize_x, scale_factor=0.5, mode='bilinear', align_corners=True)
                        x = x + resize_x
        return x, self.mid_feats


class DecoderTinyDisc(nn.Module):
    r"""
    TinyVAE without scaling
    [0,1] input [0,1] output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: Tuple[int, ...],
        block_out_channels: Tuple[int, ...],
        upsampling_scaling_factor: int,
        act_fn: str,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1),
            get_activation(act_fn),
        ]

        for i, num_block in enumerate(num_blocks):
            is_final_block = i == (len(num_blocks) - 1)
            num_channels = block_out_channels[i]

            for _ in range(num_block):
                layers.append(AutoencoderTinyBlock(num_channels, num_channels, act_fn))

            if not is_final_block:
                layers.append(nn.Upsample(scale_factor=upsampling_scaling_factor))

            conv_out_channel = num_channels if not is_final_block else out_channels
            layers.append(
                nn.Conv2d(
                    num_channels,
                    conv_out_channel,
                    kernel_size=3,
                    padding=1,
                    bias=is_final_block,
                )
            )

        self.layers = nn.Sequential(*layers)
        self.skip_nums = [1,4,8,12]
        self.gradient_checkpointing = False

    def forward(self, x: torch.FloatTensor, feats: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `DecoderTiny` class."""
        # Clamp.
        x = torch.tanh(x / 3) * 3
        logits = []
        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x, use_reentrant=False)
            else:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.layers), x)

        else:
            for i, layer in enumerate(self.layers):
                if i in self.skip_nums:
                    skip_feat = feats.pop()
                    logits.append(x)
                    logits.append(skip_feat)
                    x = x + skip_feat
                    x = layer(x)
                    logits.append(x)
                else:
                    x = layer(x)
            logits.append(x)
        return logits


class TinyDnet(ModelMixin, ConfigMixin):
    r"""
    TinyVAE without scaling
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        encoder_block_out_channels: Tuple[int, ...] = (64, 64, 64, 64),
        decoder_block_out_channels: Tuple[int, ...] = (64, 64, 64, 64),
        act_fn: str = "relu",
        latent_channels: int = 4,
        upsampling_scaling_factor: int = 2,
        num_encoder_blocks: Tuple[int, ...] = (1, 3, 3, 3),
        num_decoder_blocks: Tuple[int, ...] = (3, 3, 3, 1),
        latent_magnitude: int = 3,
        latent_shift: float = 0.5,
        force_upcast: bool = False,
        scaling_factor: float = 1.0,
    ):
        super().__init__()

        if len(encoder_block_out_channels) != len(num_encoder_blocks):
            raise ValueError("`encoder_block_out_channels` should have the same length as `num_encoder_blocks`.")
        if len(decoder_block_out_channels) != len(num_decoder_blocks):
            raise ValueError("`decoder_block_out_channels` should have the same length as `num_decoder_blocks`.")

        self.encoder = EncoderTinyDisc(
            in_channels=in_channels,
            out_channels=latent_channels,
            num_blocks=num_encoder_blocks,
            block_out_channels=encoder_block_out_channels,
            act_fn=act_fn,
        )

        self.decoder = DecoderTinyDisc(
            in_channels=latent_channels,
            out_channels=out_channels,
            num_blocks=num_decoder_blocks,
            block_out_channels=decoder_block_out_channels,
            upsampling_scaling_factor=upsampling_scaling_factor,
            act_fn=act_fn,
        )

        self.latent_magnitude = latent_magnitude
        self.latent_shift = latent_shift
        self.scaling_factor = scaling_factor

        self.use_slicing = False
        self.use_tiling = False

        # only relevant if vae tiling is enabled
        self.spatial_scale_factor = 2**out_channels
        self.tile_overlap_factor = 0.125
        self.tile_sample_min_size = 512
        self.tile_latent_min_size = self.tile_sample_min_size // self.spatial_scale_factor

        self.register_to_config(block_out_channels=decoder_block_out_channels)
        self.register_to_config(force_upcast=False)


    def forward(
        self,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, Tuple[torch.FloatTensor]]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        enc, enc_feats = self.encoder(sample)

        logits = self.decoder(enc, enc_feats)

        return logits



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





######################################
#### ref to Diffusion2GAN
#### initialized with a tinyvae encoder and decoder
#### TinyDnet
class TinyDisc(nn.Module):
    def __init__(self, disc_start,  pixelloss_weight=1.0,
                 disc_weight=1.0,
                 disc_conditional=False,
                 disc_loss="hinge", max_buff_len=10000,
                 pretrain_path="/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd/diffusion_pytorch_model.bin"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.pixel_weight = pixelloss_weight

        self.discriminator = TinyDnet()
        self.discriminator.load_state_dict(torch.load(pretrain_path), strict=True)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.rec = LPIPS().eval()
        self.buffer = []
        self.max_buff_len = max_buff_len
        self.count = 0
 
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
            logits_fake = self.discriminator(reconstructions.contiguous())
            # logits_real = self.discriminator(inputs.contiguous().detach())
            
            g_loss = 0
            for i in range(len(logits_fake)):
                g_loss = g_loss - torch.mean(logits_fake[i])
            g_loss  = g_loss / len(logits_fake)
            g_weight = 1.
            if global_step < self.discriminator_iter_start:
                g_weight = 0.0
            else:
                g_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer)


            loss = rec_loss + 0.5* g_weight* g_loss 
            log = {"total_loss":loss.clone().detach().mean(),
                   "lpips_loss": rec_loss.detach().mean(),
                   "gan_loss": (g_loss*0.5*g_weight).detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            mini_batch = reconstructions.shape[0]
            rec_device = reconstructions.device

            # sample R1 penalty
            self.count = (self.count + 1)%17
            R1 = 0
            if (self.count+1) % 16==0:
                real_data = inputs[:1].clone().requires_grad_(True)
                real_logits = self.discriminator(real_data)
                for real_logit in real_logits:
                    grad_real =  torch.autograd.grad(
                        outputs=real_logit.sum(), inputs=real_data, create_graph=True, retain_graph=True, only_inputs=True 
                    )[0]
                    R1 = R1 + grad_real.pow(2).mean()
                R1 = R1 / len(real_logits)
                

            inputs = inputs.contiguous().detach()
            reconstructions = reconstructions.contiguous().detach()
            # add to replay buffer
            """  replay buffer     
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
            """
          
            logits_real = self.discriminator(inputs)
            logits_fake = self.discriminator(reconstructions)
            d_loss = 0
            for i in range(len(logits_real)):
                d_loss = d_loss + self.disc_loss(logits_real[i], logits_fake[i])
            d_loss = d_loss / len(logits_real)


            d_loss =  d_loss +  1e-5 * R1
            if (self.count+1) % 16==0:
                log = {"disc_loss": d_loss.clone().detach().mean(),
                    "logits_real": logits_real[-1].detach().mean(),
                    "R1": 1e-5 * R1.detach(),
                    "logits_fake": logits_fake[-1].detach().mean()
                    }
            else:
                log = {"disc_loss": d_loss.clone().detach().mean(),
                    "logits_real": logits_real[-1].detach().mean(),
                    "logits_fake": logits_fake[-1].detach().mean()
                    }
            return d_loss, log


######################################
#### ref to Diffusion2GAN
#### initialized with a tinyvae encoder and decoder
#### TinyDnet  non-saturate loss + large gan loss weight
class TinyDisc2(nn.Module):
    def __init__(self, disc_start,  pixelloss_weight=1.0,
                 disc_weight=1.0,
                 disc_conditional=False,
                 disc_loss="hinge", max_buff_len=10000,
                 pretrain_path="/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd/diffusion_pytorch_model.bin"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.pixel_weight = pixelloss_weight

        self.discriminator = TinyDnet()
        self.discriminator.load_state_dict(torch.load(pretrain_path), strict=True)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.rec = LPIPS().eval()
        self.buffer = []
        self.max_buff_len = max_buff_len
        self.count = 0
 
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
            logits_fake = self.discriminator(reconstructions.contiguous())
            logits_real = self.discriminator(inputs.contiguous().detach())
            
            g_loss = 0
            for i in range(len(logits_fake)):
                g_loss = g_loss + torch.abs( (logits_fake[i] - logits_real[i]).mean())


            g_loss  = g_loss / len(logits_fake)
            g_weight = 1.
            if global_step < self.discriminator_iter_start:
                g_weight = 0.0
            else:
                g_weight = 1.0
                #g_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer)


            loss = rec_loss +  g_weight* g_loss 
            log = {"total_loss":loss.clone().detach().mean(),
                   "lpips_loss": rec_loss.detach().mean(),
                   "gan_loss": (g_loss*g_weight).detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            mini_batch = reconstructions.shape[0]
            rec_device = reconstructions.device

            # sample R1 penalty
            self.count = (self.count + 1)%17
            R1 = 0
            if (self.count+1) % 16==0:
                real_data = inputs[:1].clone().requires_grad_(True)
                real_logits = self.discriminator(real_data)
                for real_logit in real_logits:
                    grad_real =  torch.autograd.grad(
                        outputs=real_logit.sum(), inputs=real_data, create_graph=True, retain_graph=True, only_inputs=True 
                    )[0]
                    R1 = R1 + grad_real.pow(2).mean()
                R1 = R1 / len(real_logits)
                

            inputs = inputs.contiguous().detach()
            reconstructions = reconstructions.contiguous().detach()
            # add to replay buffer
            """  replay buffer     
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
            """
          
            logits_real = self.discriminator(inputs)
            logits_fake = self.discriminator(reconstructions)
            d_loss = 0
            for i in range(len(logits_real)):
                d_loss = d_loss + self.disc_loss(logits_real[i], logits_fake[i])
            d_loss = d_loss / len(logits_real)


            d_loss =  d_loss +  1e-4 * R1
            if (self.count+1) % 16==0:
                log = {"disc_loss": d_loss.clone().detach().mean(),
                    "logits_real": logits_real[-1].detach().mean(),
                    "R1": 1e-5 * R1.detach(),
                    "logits_fake": logits_fake[-1].detach().mean()
                    }
            else:
                log = {"disc_loss": d_loss.clone().detach().mean(),
                    "logits_real": logits_real[-1].detach().mean(),
                    "logits_fake": logits_fake[-1].detach().mean()
                    }
            return d_loss, log

if __name__ == "__main__":
    import pdb
    # input  = torch.rand((8,3,512,512)).cuda()
    # dec = torch.rand((8,3,512,512)).cuda()
    # encoder = EncoderTinyDisc(in_channels=3,out_channels=4,num_blocks=[1,3,3,3], block_out_channels=[64,64,64,64],act_fn="relu").cuda()
    # decoder = DecoderTinyDisc(in_channels=4,out_channels=3,num_blocks=[3,3,3,1],block_out_channels=[64,64,64,64],upsampling_scaling_factor=2,act_fn="relu").cuda()
    # out, feats = encoder(input)
    # logits = decoder(out, feats)

    # tinyDnet = TinyDnet().cuda()
    # tinyDnet.load_state_dict(torch.load("/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd/diffusion_pytorch_model.bin"), strict=True)
    # logtis =  tinyDnet(input)
    
    D_net = TinyDisc(disc_start=1).cuda()
    input  = torch.rand((8,3,512,512)).cuda()
    dec = torch.rand((8,3,512,512)).cuda()
    loss, log = D_net(input, dec, optimizer_idx = 0, global_step=25)    
    
    print(log)


