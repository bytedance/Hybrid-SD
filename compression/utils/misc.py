# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

import csv
import os
from PIL import Image
import torch
import gc

def get_file_list_from_csv(csv_file_path):
    file_list = []
    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)        
        next(csv_reader, None) # Skip the header row
        for row in csv_reader: # (row[0], row[1]) = (img name, txt prompt) 
            file_list.append(row)
    return file_list

def change_img_size(input_folder, output_folder, resz=256):
    img_list = sorted([file for file in os.listdir(input_folder) if file.endswith('.jpg')])
    for i, filename in enumerate(img_list):
        img = Image.open(os.path.join(input_folder, filename))
        img.resize((resz, resz)).save(os.path.join(output_folder, filename))
        img.close()
        if i % 2000 == 0:
            print(f"{i}/{len(img_list)} | {filename}: resize to {resz}")

def get_activation(mem, name):
    def get_output_hook(module, input, output):
        mem[name] = output

    return get_output_hook

def add_hook(net, mem, mapping_layers):
    for n, m in net.named_modules():
        if n in mapping_layers:
            m.register_forward_hook(get_activation(mem, n))

def copy_weight_from_teacher(unet_stu, unet_tea, student_type):

    connect_info = {} # connect_info['TO-student'] = 'FROM-teacher'
    if student_type in ["bk_base", "bk_small"]:
        connect_info['up_blocks.0.resnets.1.'] = 'up_blocks.0.resnets.2.'
        connect_info['up_blocks.1.resnets.1.'] = 'up_blocks.1.resnets.2.'
        connect_info['up_blocks.1.attentions.1.'] = 'up_blocks.1.attentions.2.'
        connect_info['up_blocks.2.resnets.1.'] = 'up_blocks.2.resnets.2.'
        connect_info['up_blocks.2.attentions.1.'] = 'up_blocks.2.attentions.2.'
        connect_info['up_blocks.3.resnets.1.'] = 'up_blocks.3.resnets.2.'
        connect_info['up_blocks.3.attentions.1.'] = 'up_blocks.3.attentions.2.'
    elif student_type in ["bk_tiny"]:
        connect_info['up_blocks.0.resnets.0.'] = 'up_blocks.1.resnets.0.'
        connect_info['up_blocks.0.attentions.0.'] = 'up_blocks.1.attentions.0.'
        connect_info['up_blocks.0.resnets.1.'] = 'up_blocks.1.resnets.2.'
        connect_info['up_blocks.0.attentions.1.'] = 'up_blocks.1.attentions.2.'
        connect_info['up_blocks.0.upsamplers.'] = 'up_blocks.1.upsamplers.'
        connect_info['up_blocks.1.resnets.0.'] = 'up_blocks.2.resnets.0.'
        connect_info['up_blocks.1.attentions.0.'] = 'up_blocks.2.attentions.0.'
        connect_info['up_blocks.1.resnets.1.'] = 'up_blocks.2.resnets.2.'
        connect_info['up_blocks.1.attentions.1.'] = 'up_blocks.2.attentions.2.'
        connect_info['up_blocks.1.upsamplers.'] = 'up_blocks.2.upsamplers.'
        connect_info['up_blocks.2.resnets.0.'] = 'up_blocks.3.resnets.0.'
        connect_info['up_blocks.2.attentions.0.'] = 'up_blocks.3.attentions.0.'
        connect_info['up_blocks.2.resnets.1.'] = 'up_blocks.3.resnets.2.'
        connect_info['up_blocks.2.attentions.1.'] = 'up_blocks.3.attentions.2.'       
    else:
        raise NotImplementedError

    for k in unet_stu.state_dict().keys():
        flag = 0
        k_orig = k
        for prefix_key in connect_info.keys():
            if k.startswith(prefix_key):
                flag = 1
                k_orig = k_orig.replace(prefix_key, connect_info[prefix_key])            
                break

        if flag == 1:
            print(f"** forced COPY {k_orig} -> {k}")
        else:
            print(f"normal COPY {k_orig} -> {k}")
        unet_stu.state_dict()[k].copy_(unet_tea.state_dict()[k_orig])

    return unet_stu


### update version
def getActivation(activation,name,residuals_present):
    # the hook signature
    if residuals_present:
        def hook(model, input, output):
            activation[name] = output[0]
    else:
        def hook(model, input, output):
            activation[name] = output
    return hook
    
def add_hook_update(unet, dicts, model_type, accelerator, teacher=False):
    unet=accelerator.unwrap_model(unet)
    if teacher:
        for i in range(4):
            unet.down_blocks[i].register_forward_hook(getActivation(dicts,'d'+str(i),True))
        unet.mid_block.register_forward_hook(getActivation(dicts,'m',False))
        for i in range(4):
            unet.up_blocks[i].register_forward_hook(getActivation(dicts,'u'+str(i),False))
    else:
        num_blocks= 4 if model_type=="bk_small" else 3
        for i in range(num_blocks):
            unet.down_blocks[i].register_forward_hook(getActivation(dicts,'d'+str(i),True))
        if model_type=="bk_small":
            unet.mid_block.register_forward_hook(getActivation(dicts,'m',False))
        for i in range(num_blocks):
            unet.up_blocks[i].register_forward_hook(getActivation(dicts,'u'+str(i),False))
        

def prepare_unet(unet, model_type, accelerator):
    unet=accelerator.unwrap_model(unet)
    assert model_type in ["bk_tiny", "bk_small"]
    # Set mid block to None if mode is other than base
    if model_type != "bk_small":
        unet.mid_block = None
        
    # Commence deletion of resnets/attentions inside the U-net
    # Handle Down Blocks
    for i in range(3):
        delattr(unet.down_blocks[i].resnets, "1")
        delattr(unet.down_blocks[i].attentions, "1")

    if model_type == "bk_tiny":
        delattr(unet.down_blocks, "3")
        unet.down_blocks[2].downsamplers = None

    else:
        delattr(unet.down_blocks[3].resnets, "1")
    # Handle Up blocks
    unet.up_blocks[0].resnets[1] = unet.up_blocks[0].resnets[2]
    delattr(unet.up_blocks[0].resnets, "2")
    for i in range(1, 4):
        unet.up_blocks[i].resnets[1] = unet.up_blocks[i].resnets[2]
        unet.up_blocks[i].attentions[1] = unet.up_blocks[i].attentions[2]
        delattr(unet.up_blocks[i].attentions, "2")
        delattr(unet.up_blocks[i].resnets, "2")
    if model_type == "bk_tiny":
        for i in range(3):
            unet.up_blocks[i] = unet.up_blocks[i + 1]
        delattr(unet.up_blocks, "3")
    torch.cuda.empty_cache()
    gc.collect()