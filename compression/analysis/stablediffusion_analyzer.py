import torch.nn as nn 
from .basemodel_analyzer import BaseModelAnalyzer
from collections import OrderedDict
import pickle
from pathlib import Path

class StableDiffusionAnalyzer(BaseModelAnalyzer):
    def __init__(self, model, model_name="StableDiffusion v1.5", save_path="./", format="pth"):
        super().__init__(model, save_path, format)
        self.model_name = model_name

    def collect_weight(self):
        weight_dict = OrderedDict()
        path_weight_dict =  Path(f'{self.output_dir}/weights.pkl')
        if path_weight_dict.exists():
            with open(path_weight_dict, 'rb') as f:
                weight_dict = pickle.load(f)

            return weight_dict

        for module_id, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear, nn.Embedding)):
                print(module_id, name)
                weight_dict[name] = {"data": module.weight.detach().cpu().data, "class_name": module._get_name()}

        with open(path_weight_dict, 'wb') as f:
            pickle.dump(weight_dict, f)

        return weight_dict
