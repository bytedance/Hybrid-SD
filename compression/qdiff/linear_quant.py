import torch
from torch import nn
import logging
from .quant_layer import UniformAffineQuantizer, round_ste, floor_ste
import numpy as np

logger = logging.getLogger(__name__)

def _dequantize_linear(x_int, scale, bias, axis=0):
    if len(x_int.shape) == 1:  # vector situation, treat as 1 channel
        x_int = x_int.reshape((1, x_int.shape[0]))

    rank = len(x_int.shape)
    if axis == 1:
        transposed_axis_order = (1, 0) + tuple(range(2, rank))
        x_int = np.transpose(x_int, transposed_axis_order)

    num_channels = x_int.shape[0]
    broadcast_shape = (num_channels,) + (1,) * (rank - 1)
    scale = scale.reshape(broadcast_shape)
    bias = bias.reshape(broadcast_shape)
    weight = x_int.astype("float") * scale + bias
    if axis == 1:
        weight = np.transpose(weight, transposed_axis_order)

    return weight

class LinearQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor):
        super(LinearQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.axis = 0

    def forward(self, x):
        if len(x.shape) == 1:  # vector situation, treat as 1 channel
            x = x.reshape((1, x.shape[0]))
        rank = len(x.shape)
        if self.axis == 1:
            transposed_axis_order = (1, 0) + tuple(range(2, rank))
            x = np.transpose(x, transposed_axis_order)

        num_channels = weight.shape[0]
        shape = weight.shape
        weight = weight.reshape((num_channels, -1))  # [C, L]

        a = np.amin(weight, axis=-1)  # [C,]
        b = np.amax(weight, axis=-1)  # [C,]

        if self.sym:
            r = np.maximum(np.abs(a), np.abs(b)) 
            scale = r / ((1 << self.n_bits) / 2.0 - 1)
            bias = -(1 << self.n_bits) / 2.0 * scale
            num = weight - bias[:, None]
            denom = scale[:, None]
            x_int = np.divide(
                num, denom, out=np.zeros_like(num), where=(np.abs(denom) > 1e-6)
            )
            x_int = np.round(x_int)
        else:
            qb = (1 << self.nbits) - 1
            scale = (b - a) / qb
            inv_scale = np.divide(
                1.0, scale, out=np.zeros_like(scale), where=(np.abs(scale) > 1e-6)
            )
            bias = a
            x_int = (weight - a[:, None]) * inv_scale[:, None]
            x_int = np.round(x_int)

        # Reshape
        x_int = x_int.reshape(shape)
        if self.axis == 1:
            x_int = np.transpose(x_int, transposed_axis_order)

        if len(x_int.shape) == 1:  # vector situation, treat as 1 channel
            x_int = x_int.reshape((1, x_int.shape[0]))

        rank = len(x_int.shape)
        if self.axis == 1:
            transposed_axis_order = (1, 0) + tuple(range(2, rank))
            x_int = np.transpose(x_int, transposed_axis_order)

        num_channels = x_int.shape[0]
        broadcast_shape = (num_channels,) + (1,) * (rank - 1)
        scale = scale.reshape(broadcast_shape)
        bias = bias.reshape(broadcast_shape)
        x_float_q = x_int.astype("float") * scale + bias
        if self.axis == 1:
            x_float_q = np.transpose(x_float_q, transposed_axis_order)

        return x_float_q

    def extra_repr(self):
        s = 'bit={n_bits}, symmetric={sym}' 
        return s.format(**self.__dict__)
