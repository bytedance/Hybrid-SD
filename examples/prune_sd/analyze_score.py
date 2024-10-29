# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import numpy as np 
from compression.prune_sd.prune_utils import calculate_score
import pickle

result_dir = Path('results/NaivePrune/bk-sdm-tiny/prune_oneshot')
layer_info_path = result_dir / 'layer_infos.pickle'

with open(layer_info_path,'rb') as f:
    all_layer_info = pickle.load(f)

eval_latents, keys = [], []
for layer_info in all_layer_info:
    prune_type = layer_info['prune_type']
    vals = prune_type.split('_')
    layer_name = ('_').join(vals[:-1])
    ratio = vals[-1]
    if layer_name == 'baseline' or ratio in ['0.5']:
        print(f"get latent of {prune_type}")
        eval_latents.append(layer_info['np_latents'])
        if layer_name != 'baseline':
            keys.append(layer_name)

scores = calculate_score(eval_latents)
score_dict = {}
for k, score in zip(keys, scores):
    score_dict[k] = score
print(score_dict)

path = result_dir / 'score.pkl'
with open(path, 'wb') as f:
    pickle.dump(score_dict, f)
print(f"save score to {path}")

