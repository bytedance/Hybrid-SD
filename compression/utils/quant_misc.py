import torch
from collections import defaultdict
ckpt_path = ''
ckpt = torch.load(ckpt_path)

params_counter_dict = dict()
quant_dict = defaultdict(list)

for k, v in ckpt.items():
    params_counter_dict[k] = {'shape': v.shape, 'size': v.shape.numel()}
    import pdb; pdb.set_trace()

    if 'weight_quantizer' in k:
        prefix = k.index('weight_quantizer')
        postfix_index = prefix + len('weight_quantizer')
        prefix_name = k[:prefix]
        quant_dict[prefix_name].append(v.numel())

print('original params size:')

print('quant params size:')
    