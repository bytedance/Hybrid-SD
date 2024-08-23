import torch
import torch.nn as nn 
import os 
from abc import ABC
from collections import OrderedDict
from . import plot_utils as plot_utils
from pathlib import Path
import pickle
import pandas as pd

class BaseModelAnalyzer(ABC):
    def __init__(self, model, output_dir="./", format="pth", save_plots=True):
        super().__init__()
        self.model = model
        self.step = 0 
        self.act_hooks = []
        self.output_dir = output_dir
        self.plot_imgs_path = os.path.join(self.output_dir, "images")
        os.makedirs(self.plot_imgs_path, exist_ok=True)
        self.act_format = format
        self.weight_dict = self.collect_weight()
        self.act_dict = OrderedDict()
        self.other_layers = [] 
        self.block_layers = []
        self.all_block_layers = []
        self.save_plots = save_plots

    def collect_weight(self):
        weight_dict = OrderedDict()
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
                weight_dict[name] = module.weight.detach().cpu().data
        return weight_dict

    def plot_weights_boxplot(self, prefix=None, figsize=(16, 8)):
        print("Plotting weights boxplot...")
        weights_boxplot_path = os.path.join(self.plot_imgs_path, "weights_boxplot.png")
        if prefix is not None:
            weight_dict = {}
            for k, v in self.weight_dict.items():
                for p in prefix:
                    if not k.startswith(p):
                        continue
                    weight_dict[k] = v
        else:
            weight_dict = self.weight_dict
        plot_utils.layer_boxplot(weights_boxplot_path, weight_dict, title=self.model_name, figsize=figsize)
        print(f"Save weights boxplot to {weights_boxplot_path}")
                

    def plot_weights_3d_dist(self,  prefix=None, skip_generated=True, figsize=(32, 16)):
        print("Plotting weights 3d distribution...")
        dist_dir = os.path.join(self.plot_imgs_path, "threedist") 
        os.makedirs(dist_dir, exist_ok=True)
        if prefix is not None:
            weight_dict = {}
            for k, v in self.weight_dict.items():
                for p in prefix:
                    if not k.startswith(p):
                        continue
                    weight_dict[k] = v
        else:
            weight_dict = self.weight_dict

        for name, item in weight_dict.items():
            output_dir = os.path.join(dist_dir, f"{name}.png")
            if 'resnet' not in name:
                continue
            
            if skip_generated and Path(output_dir).exists():
                continue

            print(f"plotting {name} weights")
            plot_utils.plot_3d_dist_histogram(output_dir, item['data'], name)
            

    def plot_weight_2d_dist(self, prefix=None):
        print("Plotting weights 2d distribution...")
        dist_dir = os.path.join(self.plot_imgs_path, "twodist") 
        os.makedirs(dist_dir, exist_ok=True)
        for name, weight in self.weight_dict.items():
            output_dir = os.path.join(dist_dir, f"{name}.png")
            if prefix is not None and not name.startswith(prefix):
                continue
            plot_utils.plot_dist_histogram(output_dir, weight)
            

    def save_info(self):
        info = []
        for name, weight in self.weight_dict.items():
            print(name, weight.shape)
            row = {
                'name': name,
                'shape': weight.size()
            }
            info.append(row)
            
        df = pd.DataFrame(info)
        df_path = Path(self.output_dir) / 'model_info.csv'
        df.to_csv(df_path, index=False)
        print(f"Save model info to {df_path}")


    
