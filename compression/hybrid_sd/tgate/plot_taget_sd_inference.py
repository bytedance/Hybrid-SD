import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from compression.hybrid_sd.tgate.SD_tgate import TgateSDLoader
from compression.prune_sd.models.unet_2d_condition import UNet2DConditionModel as CustomUNet2DConditionModel
from diffusers import PixArtAlphaPipeline,StableDiffusionXLPipeline,StableDiffusionPipeline
from diffusers import UNet2DConditionModel, LCMScheduler
from diffusers import DPMSolverMultistepScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of TGATE.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="the input prompts",
    )
    parser.add_argument(
        "--saved_path",
        type=str,
        default=None,
        required=True,
        help="Path to save the generated results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='sd_1.5',
        help="[pixart,sd_xl,sd_2.1,sd_1.5,lcm_sdxl,lcm_pixart,bk_sdm]",
    )
    parser.add_argument(
        "--gate_step",
        type=int,
        default=10,
        help="When re-using the cross-attention",
    )
    parser.add_argument(
        "--inference_step",
        type=int,
        default=25,
        help="total inference steps",
    )
    parser.add_argument(
        '--deepcache', 
        action='store_true', 
        default=False, 
        help='do deep cache',
    )
    
    args = parser.parse_args()
    return args


"""
python3  compression/hybrid_sd/tgate/taget_sd_inference.py  --model sd_1.5 --prompt  "A yellow taxi cab sitting below tall buildings" --saved_path "."
    
python3  compression/hybrid_sd/tgate/taget_sd_inference.py  --model bk_sdm --prompt  "A yellow taxi cab sitting below tall buildings" --saved_path "."


python3  compression/hybrid_sd/tgate/plot_taget_sd_inference.py  --model sd_1.5 --prompt  "A yellow taxi cab sitting below tall buildings" --saved_path "results/tgate/"  --gate_step 10

python3  compression/hybrid_sd/tgate/plot_taget_sd_inference.py  --model sd_1.4  --saved_path "results/tgate/"  --gate_step 30

"""



def calculate_stats(tensor_list):
    tensors = torch.stack(tensor_list)
    mean = torch.mean(tensors).cpu().numpy()
    var = torch.var(tensors).cpu().numpy()
    return mean, var

def inspect_attention_modules(model):
    attention_modules = []
    diffs = {str(i): [] for i in range(2, 26)}
    
    for name, module in model.named_modules():
        if 'Attention' in module.__class__.__name__:
            for key in module.diffs.keys():
                diffs[key].append(module.diffs[key])
            attention_modules.append((name, module))
    
    # get mean value
    # for key in diffs.keys():
    #     diffs[key] = torch.stack(diffs[key]).mean()
    return diffs

def update_global_diffs(global_diffs, diffs):
    #diffs = {str(i): [] for i in range(2, 26)}
    for key in diffs.keys():
        global_diffs[key] = global_diffs[key] + diffs[key]
    return global_diffs



def plot_mean_var_time(diffs):
    time_steps = []
    means = []
    variances = []
    for time, tensor_list in diffs.items():
        mean, var = calculate_stats(tensor_list)
        time_steps.append(int(time))
        means.append(mean)
        variances.append(var)
        
        
    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot mean line
    plt.plot(time_steps, means, 'k-', linewidth=2)

    # Plot shaded area for variance
    plt.fill_between(time_steps, np.array(means) - np.array(variances), np.array(means) + np.array(variances), 
                    alpha=0.3, color='blue')

    # Customize the plot
    plt.title('Difference of Cross-attention Maps Between Consecutive Inference Steps')
    plt.xlabel('Inference Steps')
    plt.ylabel('Cross-attention Difference')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 25)
    #plt.ylim(-0.0005, 0.0020)  # Set y-axis limit with some padding
    #plt.ylim(plt.ylim(-0.0005, max(np.array(means) + np.array(variances)) * 1.1) )

    # Add color transition
    # plt.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
    plt.fill_between(time_steps, np.array(means) - np.array(variances), np.array(means) + np.array(variances), 
                    where=(np.array(time_steps) >= 0), alpha=0.3, color='red')

    plt.tight_layout()
    plt.show()
    # Save the figure
    plt.savefig('results/tgate/cross_attention_difference.png', dpi=300, bbox_inches='tight')
        
    # Plotting
    plt.figure(figsize=(12, 6))

    plt.errorbar(time_steps, means, yerr=variances, fmt='o-', capsize=5, capthick=2, ecolor='red', color='blue')
    plt.title('Mean Values with Standard Deviation Error Bars')
    plt.xlabel('Time Step')
    plt.ylabel('Mean Value')
    plt.grid(True)

    # Add a legend
    plt.fill_between([], [], color='red', alpha=0.3, label='Standard Deviation')
    plt.plot([], [], 'o-', color='blue', label='Mean')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the figure
    plt.savefig('results/tgate/module_diffs_stats_with_error_bars.png')

def cocolist():
    json_file = '/mnt/bn/bytenn-yg2/datasets/coco2017_val/annotations/captions_val2017.json'
    train_path = '/mnt/bn/bytenn-yg2/datasets/coco2017_val/imgs'
    img_list=[]
    coco=COCO(json_file)
    img_ids=coco.getImgIds()
    
    for img_id in img_ids:
        img = coco.loadImgs(img_id)[0]
        image_file = f'{train_path}/{img["file_name"]}'
        ann_ids = coco.getAnnIds(imgIds=img_id)
        captions = coco.loadAnns(ann_ids)
        text=[cap['caption'] for cap in captions][0]
        tmp={}
        tmp['img']=image_file
        tmp['caption']=text
        img_list.append(tmp)
    return img_list



if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.saved_path, exist_ok=True)
    saved_path = os.path.join(args.saved_path, 'test.png')
    
    global_diffs = {str(i): [] for i in range(2, 26)}
    
    unet = None
    global_mean = {str(i): [] for i in range(2, 26)}
    global_var = {str(i): [] for i in range(2, 26)}
    if args.model in ['sd_2.1', 'sd_1.5', 'sd_1.4', 'bk_sdm']:
        if args.model == 'sd_1.5':
            repo_id = "/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5"
        elif args.model == 'sd_2.1':
            repo_id = "stabilityai/stable-diffusion-2-1"
        elif args.model == 'sd_1.4':
            repo_id = "/mnt/bn/bytenn-yg2/pretrained_models/CompVis--stable-diffusion-v1-4"
        elif args.model == 'bk_sdm':
            repo_id = "/mnt/bn/bytenn-yg2/pretrained_models/nota-ai--bk-sdm-tiny"
            # unet = CustomUNet2DConditionModel.from_pretrained(
            #     repo_id, subfolder="unet_target" #, revision=args.non_ema_revision
            # )
        print(f"====== inference with {repo_id} ======")
        pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        
        pipe = TgateSDLoader(pipe)
        pipe = pipe.to("cuda")


        img_list=cocolist()
        prompts = [data['caption'] for data in img_list][:5000]
        #names= [data['img'] for data in img_list][:5000]
        
        batch_size = 50
        print("batch_size is :", batch_size)
        
        for i in range(0,len(prompts)//(batch_size)):
            prompt = prompts[i*batch_size: (i+1)*batch_size]
            image = pipe.tgate(prompt,
                            num_inference_steps=args.inference_step,
                            guidance_scale=7.5,
                            gate_step=args.gate_step,
                            ).images[0]
            
            diffs = inspect_attention_modules(pipe.unet)
            global_diffs = update_global_diffs(global_diffs, diffs)

        
        torch.save(global_diffs, 'results/tgate/diffs.pt')
        
        global_diffs = torch.load('results/tgate/diffs.pt')

        plot_mean_var_time(global_diffs)


    #image.save(saved_path)