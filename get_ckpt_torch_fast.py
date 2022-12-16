# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Merge model parallel partitions."""

import os
import random
import sys
import argparse
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

def main():
    state_dict_path = "/checkpoints/codegeex/codegeex_13b.pt"
    print("Loading state dict ...")
    sd = torch.load(state_dict_path, map_location="cpu")
    
    for i in range(40):
        if i < 39:
            query_weight = sd['module']['language_model']['transformer'].pop(f'layers.{i}.attention.query.weight', None)
            query_bias = sd['module']['language_model']['transformer'].pop(f'layers.{i}.attention.query.bias', None)
            key_weight = sd['module']['language_model']['transformer'].pop(f'layers.{i}.attention.key.weight', None)
            key_bias = sd['module']['language_model']['transformer'].pop(f'layers.{i}.attention.key.bias', None)
            value_weight = sd['module']['language_model']['transformer'].pop(f'layers.{i}.attention.value.weight', None)
            value_bias = sd['module']['language_model']['transformer'].pop(f'layers.{i}.attention.value.bias', None)
            qkv_weight = torch.cat([query_weight, key_weight, value_weight], dim=0)
            qkv_bias = torch.cat([query_bias, key_bias, value_bias])
            sd['module']['language_model']['transformer'][f'layers.{i}.attention.query_key_value.weight'] = qkv_weight
            sd['module']['language_model']['transformer'][f'layers.{i}.attention.query_key_value.bias'] = qkv_bias
        else:
            tq_key_weight = sd['module']['language_model']['transformer'].pop('topQueryLayer.attention.key.weight', None)
            tq_key_bias = sd['module']['language_model']['transformer'].pop('topQueryLayer.attention.key.bias', None)
            tq_value_weight = sd['module']['language_model']['transformer'].pop('topQueryLayer.attention.value.weight', None)
            tq_value_bias = sd['module']['language_model']['transformer'].pop('topQueryLayer.attention.value.bias', None)
            tq_kv_weight = torch.cat([tq_key_weight, tq_value_weight], dim=0)
            tq_kv_bias = torch.cat([tq_key_bias, tq_value_bias])
            sd['module']['language_model']['transformer']['topQueryLayer.attention.key_value.weight'] = tq_kv_weight
            sd['module']['language_model']['transformer']['topQueryLayer.attention.key_value.bias'] = tq_kv_bias
    
    save_ckpt_path = "/checkpoints/codegeex/codegeex_13b_ft.pt"
    torch.save(sd, save_ckpt_path)
    
if __name__ == '__main__':
    main()
