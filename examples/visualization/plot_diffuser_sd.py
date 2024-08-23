
import torch
from pytorch_lightning import seed_everything
from diffusers import StableDiffusionPipeline
from compression.analysis.stablediffusion_analyzer import StableDiffusionAnalyzer
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained("nota-ai/bk-sdm-small", subfolder="unet", torch_dtype=torch.float16).to("cuda")
output_dir = "results/bk-sdm-small/plots"
analyzer = StableDiffusionAnalyzer(unet, "bk-sdm-small", output_dir, format="pth")

# plot weight distribution
prefix = [f'down_blocks.{i}.resnets.0.conv{j}' for i in range(4) for j in range(1, 3)] + [f'up_blocks.{i}.resnets.0.conv{j}' for i in range(4) for j in range(1, 3)] + [f'up_blocks.{i}.resnets.1.conv{j}' for i in range(4) for j in range(1 ,3)]
analyzer.plot_weights_boxplot(prefix=prefix, figsize=(50, 16))
# analyzer.plot_weights_3d_dist(figsize=(50, 16))
