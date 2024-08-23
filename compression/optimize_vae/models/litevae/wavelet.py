# git clone https://github.com/fbcotter/pytorch_wavelets.git
# pip install -e /mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/publics/pytorch_wavelets
# pip install PyWavelets

import torch
import torch.nn as nn
import sys
sys.path.append("/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/publics")
from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)
from PIL import Image
from torchvision import transforms
from einops import rearrange

import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel

class DWTForwardL(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.J = J
        self.mode = mode

    def composev1(self, Yl, Yh):
        img2 = Yh[:, :, 0, :, :]
        img3 = Yh[:, :, 1, :, :]
        img4 = Yh[:, :, 2, :, :]
        top_row = torch.cat((Yl, img2), dim=3)
        bottom_row = torch.cat((img3, img4), dim=3)
        result = torch.cat((top_row, bottom_row), dim=2)
        return result
    
    def composev2(self, Yl, Yh):
        result = torch.cat((Yl.unsqueeze(dim=2), Yh), dim=2)
        result = rearrange(result, 'b c d h w -> b (c d) h w')
        return result

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        ll = x
        out_list = []
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = lowlevel.AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
            
            out_list.append(self.composev2(ll, high))
        

        return out_list


class DWTH(nn.Module):
    """ 
    return DWT high frequence components    
    """
    def __init__(self, J=1, wave='db1', mode='zero'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        self.J = J
        self.mode = mode


    def forward(self, x):
        """ Forward pass of the DWT.
        """
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            ll, high = lowlevel.AFB2D.apply(
                ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row, mode)
        
        return high



xfm = DWTForward(J=2, mode='zero', wave='haar')  
ifm = DWTInverse(mode='zero', wave='haar')
"""
X = torch.randn(10,5,64,64)
Yl, Yh = xfm(X)
# J=2 , divide h, w -> h/2, w/2
print(Yl.shape)
Y = ifm((Yl, Yh))
import numpy as np
np.testing.assert_array_almost_equal(Y.cpu().numpy(), X.cpu().numpy())
print("relative error", ((X-Y)**2).sum())
"""

def wavelet_func():
    # Load the image
    image = Image.open("imgs/gt2.png")
    # Convert the image to a PyTorch tensor
    tensor_image = transforms.ToTensor()(image)
    tensor_image = tensor_image.unsqueeze(0)
    Yl, Yh = xfm(tensor_image)


    pil_image = transforms.ToPILImage()(Yl.squeeze())

    # Save the image
    pil_image.save("imgs/gt2waveyl.jpg")

    tensor_image = ifm((Yl, Yh))
    # Convert the tensor back to a PIL image
    tensor_image = tensor_image.squeeze()
    pil_image = transforms.ToPILImage()(tensor_image)

    # Save the image
    pil_image.save("imgs/gt2wave.jpg")

def multi_wavelet_func():
    # Load the image
    image = Image.open("imgs/gt2.png")
    # Convert the image to a PyTorch tensor
    tensor_image = transforms.ToTensor()(image)
    tensor_image = tensor_image.unsqueeze(0)
    Yl, Yh = xfm(tensor_image)
    import pdb;pdb.set_trace()
    Yl1, Yh1 = xfm(Yl)

    Yl2, Yh2 = xfm(Yl1)
    B,C,H,W = tensor_image.shape
    
    # [b,3(c),h/2,w/2] [b,3(c),3,h/2,w/2]-> [b,3,h,w]

    # B,C,H,W
    img1 = Yl
    img2 = Yh[0][:, :, 0, :, :]
    img3 = Yh[0][:, :, 1, :, :]
    img4 = Yh[0][:, :, 2, :, :]

    top_row = torch.cat((img1, img2), dim=3)  # Shape [b, c, h, 2w]
    bottom_row = torch.cat((img3, img4), dim=3)  # Shape [b, c, h, 2w]

    # Concatenate along width
    result = torch.cat((top_row, bottom_row), dim=2)

    print(result.shape)
    pil0 = transforms.ToPILImage()(result.squeeze())

    pil0.save("imgs/pil0.png")

class Multi_wavelet(nn.Module):
    def __init__(self, J=3):
        super().__init__()
        self.dwt = DWTForwardL(J=J, mode='zero', wave='haar') 
    def forward(self, x):
        outs = self.dwt(x)
        return outs

    

#multi_wavelet_func()

# multi_wavelet()

if __name__ =="__main__":
    x = torch.randn(4,3,512,512)
    dwt = Multi_wavelet()
    y = dwt(x)
    print(y[0].shape, y[1].shape, y[2].shape)
    # from compression.prune_sd.calflops import calculate_flops
    # flops, macs, params = calculate_flops(
    #     model=Multi_wavelet, 
    #     input_shape=(1, 3, 512, 512),
    #     output_as_string=True,
    #     output_precision=4,
    #     print_detailed=False,
    #     print_results=False)
    # print(f'#Params={params}, FLOPs={flops}, MACs={macs}')