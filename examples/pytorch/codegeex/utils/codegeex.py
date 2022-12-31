# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import print_function

import os
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

def quant_weight(weight, dtype):
    weight = weight.cuda()
    assert dtype == 'int8'
    quantization_bit_width = 8
    weight_scale = (weight.abs().max(dim=-1).values / ((2 ** (quantization_bit_width - 1)) - 1)).half()
    weight = torch.round(weight / weight_scale[:, None]).to(torch.int8)
    return weight.cpu(), weight_scale.cpu()


class CODEGEEXWeights(object):
    def __init__(self, head_num, size_per_head, layer_num, vocab_size, max_seq_len, tensor_para_size, pipeline_para_size, dtype, int8_mode=0):
        assert (head_num % tensor_para_size == 0)

        if int8_mode != 0:
            self.weight_transpose_calibrate_quantize = torch.ops.fastertransformer.weight_transpose_calibrate_quantize

        self.head_num = head_num
        self.size_per_head = size_per_head
        self.layer_num = layer_num
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.layers_per_device = layer_num // pipeline_para_size

        local_head_num = head_num // tensor_para_size
        global_head_num = head_num
        local_hidden_units = local_head_num * size_per_head
        global_hidden_units = global_head_num * size_per_head
        local_inter_size = local_hidden_units * 4

        self.local_head_num = local_head_num
        self.global_head_num = global_head_num
        self.local_hidden_units = local_hidden_units
        self.global_hidden_units = global_hidden_units
        self.local_inter_size = local_inter_size

        self.dtype = dtype
        self.int8_mode = int8_mode

        self.w = []
        self.int8_w = []
        self.weight = []
        self.scale = []

        # Transformer blocks
        self.w.extend([torch.zeros(global_hidden_units,dtype = torch.float16)] * (layer_num-1))   # self_layernorm_gamma
        self.w.extend([torch.zeros(global_hidden_units,dtype = torch.float16)] * (layer_num-1))   # self_layernorm_beta
        self.w.extend([torch.zeros(local_hidden_units * 3, dtype = torch.float16)] * (layer_num-1))   # self_bias
        self.w.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] * (layer_num-1) )   # self_output_bias
        self.w.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] * (layer_num-1) )   # ffn_layernorm_gamma
        self.w.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] * (layer_num-1) )   # ffn_layernorm_beta
        self.w.extend([torch.zeros(local_inter_size, dtype = torch.float16)] * (layer_num-1) )   # ffn_bias1
        self.w.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] * (layer_num-1))   # ffn_bias2


        if dtype in ['fp16', 'int8']:
            w_type = torch.int8 if dtype == 'int8' else torch.float16
            self.weight.extend([torch.zeros(global_hidden_units, local_hidden_units * 3, dtype = w_type)] * (layer_num-1))   # self_kernel
            self.weight.extend([torch.zeros(local_hidden_units, global_hidden_units, dtype = w_type)] * (layer_num-1))   # self_output_kernel
            self.weight.extend([torch.zeros(global_hidden_units, local_inter_size, dtype = w_type)] * (layer_num-1))   # ffn_kernel1
            self.weight.extend([torch.zeros(local_inter_size, global_hidden_units, dtype = w_type)] * (layer_num-1))   # ffn_kernel2
        else:
            w_type = torch.int8
            self.weight.extend([torch.zeros(global_hidden_units, local_hidden_units * 3 // 2, dtype = w_type)] * (layer_num-1))   # self_kernel
            self.weight.extend([torch.zeros(local_hidden_units, global_hidden_units // 2, dtype = w_type)] * (layer_num-1))   # self_output_kernel
            self.weight.extend([torch.zeros(global_hidden_units, local_inter_size // 2, dtype = w_type)] * (layer_num-1))   # ffn_kernel1
            self.weight.extend([torch.zeros(local_inter_size, global_hidden_units // 2, dtype = w_type)] * (layer_num-1))   # ffn_kernel2

        # scale
        if dtype in ['int8', 'int4']:
            w_type = torch.float16
            self.scale.extend([torch.zeros(local_hidden_units * 3, dtype = w_type)] * (layer_num-1))   # self_kernel
            self.scale.extend([torch.zeros(global_hidden_units, dtype = w_type)] * (layer_num-1))   # self_output_kernel
            self.scale.extend([torch.zeros(local_inter_size, dtype = w_type)] * (layer_num-1))   # ffn_kernel1
            self.scale.extend([torch.zeros(global_hidden_units, dtype = w_type)] * (layer_num-1))   # ffn_kernel2
        

        # top layer
        self.w.extend([torch.zeros(global_hidden_units,dtype = torch.float16)] )   # self_layernorm_gamma
        self.w.extend([torch.zeros(global_hidden_units,dtype = torch.float16)] )   # self_layernorm_beta
        self.w.extend([torch.zeros(local_hidden_units * 1, dtype = torch.float16)]  )   # self_bias
        self.w.extend([torch.zeros(local_hidden_units * 2, dtype = torch.float16)]  )   # self_bias
        self.w.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] )   # self_output_bias
        self.w.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] )   # ffn_layernorm_gamma
        self.w.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] )   # ffn_layernorm_beta
        self.w.extend([torch.zeros(local_inter_size, dtype = torch.float16)] )   # ffn_bias1
        self.w.extend([torch.zeros(global_hidden_units, dtype = torch.float16)] )   # ffn_bias2


        if dtype in ['fp16', 'int8']:
            w_type = torch.int8 if dtype == 'int8' else torch.float16
            self.weight.extend([torch.zeros(global_hidden_units, local_hidden_units * 1, dtype = w_type)] )   # self_kernel
            self.weight.extend([torch.zeros(global_hidden_units, local_hidden_units * 2, dtype = w_type)] )   # self_kernel
            self.weight.extend([torch.zeros(local_hidden_units, global_hidden_units, dtype = w_type)] )   # self_output_kernel
            self.weight.extend([torch.zeros(global_hidden_units, local_inter_size, dtype = w_type)] )   # ffn_kernel1
            self.weight.extend([torch.zeros(local_inter_size, global_hidden_units, dtype = w_type)] )   # ffn_kernel2
        else:
            w_type = torch.int8
            self.weight.extend([torch.zeros(global_hidden_units, local_hidden_units * 1 // 2, dtype = w_type)] )   # self_kernel
            self.weight.extend([torch.zeros(global_hidden_units, local_hidden_units * 2 // 2, dtype = w_type)] )   # self_kernel
            self.weight.extend([torch.zeros(local_hidden_units, global_hidden_units // 2, dtype = w_type)] )   # self_output_kernel
            self.weight.extend([torch.zeros(global_hidden_units, local_inter_size // 2, dtype = w_type)] )   # ffn_kernel1
            self.weight.extend([torch.zeros(local_inter_size, global_hidden_units // 2, dtype = w_type)] )   # ffn_kernel2

        # scale
        if dtype in ['int8', 'int4']:
            w_type = torch.float16
            self.scale.extend([torch.zeros(local_hidden_units * 1, dtype = w_type)] )   # self_kernel
            self.scale.extend([torch.zeros(local_hidden_units * 2, dtype = w_type)] )   # self_kernel
            self.scale.extend([torch.zeros(global_hidden_units, dtype = w_type)] )   # self_output_kernel
            self.scale.extend([torch.zeros(local_inter_size, dtype = w_type)] )   # ffn_kernel1
            self.scale.extend([torch.zeros(global_hidden_units, dtype = w_type)] )   # ffn_kernel2

        # After Transformer blocks
        self.w.append(torch.zeros(global_hidden_units, dtype = torch.float16))   # layernorm_gamma
        self.w.append(torch.zeros(global_hidden_units, dtype = torch.float16))   # layernorm_beta
        self.w.append(torch.zeros(max_seq_len, global_hidden_units, dtype = torch.float16))   # position_encoding_table
        self.w.append(torch.zeros(vocab_size, global_hidden_units, dtype = torch.float16))   # embedding_table
        self.w.append(torch.zeros(vocab_size, global_hidden_units, dtype = torch.float16))   # embedding_kernel
        self.w.append(torch.zeros(max_seq_len, global_hidden_units, dtype = torch.float16))   # topquery embedding
        # Initialization
        #self._map(lambda w: torch.nn.init.normal_(w, mean=0., std=1.))

        if (self.int8_mode != 0):
            self.int8_w.extend([torch.zeros(global_hidden_units, local_hidden_units *
                               3, dtype=torch.int8)] * layer_num)   # self_int8_kernel
            self.scale.extend([torch.zeros(local_hidden_units * 3, dtype=torch.float)] * layer_num)   # self_scale
            self.int8_w.extend([torch.zeros(local_hidden_units, global_hidden_units, dtype=torch.int8)]
                               * layer_num)   # self_output_int8_kernel
            self.scale.extend([torch.zeros(global_hidden_units, dtype=torch.float)] * layer_num)   # self_output_scale
            self.int8_w.extend([torch.zeros(global_hidden_units, local_inter_size,
                               dtype=torch.int8)] * layer_num)   # ffn_int8_kernel1
            self.scale.extend([torch.zeros(local_inter_size, dtype=torch.float)] * layer_num)   # ffn_scale1
            self.int8_w.extend([torch.zeros(local_inter_size, global_hidden_units,
                               dtype=torch.int8)] * layer_num)   # ffn_int8_kernel2
            self.scale.extend([torch.zeros(global_hidden_units, dtype=torch.float)] * layer_num)   # ffn_scale2

    def __getitem__(self, idx):
        return self.w[idx]

    def __setitem__(self, idx, val):
        self.w[idx] = val

    def __len__(self):
        return len(self.w)

    def _map(self, func):
        for w in [self.w, self.weight, self.scale]:
            for i in range(len(w)):
                if isinstance(w[i], list):
                    for j in range(len(w[i])):
                        w[i][j] = func(w[i][j])
                else:
                    w[i] = func(w[i])

    def _map_int8(self, func):
        for i in range(len(self.int8_w)):
            if isinstance(self.int8_w[i], list):
                for j in range(len(self.int8_w[i])):
                    self.int8_w[i][j] = func(self.int8_w[i][j])

            else:
                self.int8_w[i] = func(self.int8_w[i])
        for i in range(len(self.scale)):
            if isinstance(self.scale[i], list):
                for j in range(len(self.scale[i])):
                    self.scale[i][j] = func(self.scale[i][j])

            else:
                self.scale[i] = func(self.scale[i])

    def load(self, ckpt_path, tensor_para_rank, pipeline_para_rank):
        if not os.path.exists(ckpt_path):
            return False

        # checkpoint_name = os.path.join(ckpt_path, 'mp_rank_{:02d}_model_states.pt'.format(tensor_para_rank))

        module = torch.load(ckpt_path, map_location='cpu')['module']['language_model']

        # Load
        tensor_model_parallel_size = self.tensor_para_size
        layer_num = self.layer_num

        w = []
        weight = []
        scale = []
        # Load
        num_splits = 3

        hidden_dim, local_dim = module['transformer']['layers.0.attention.query_key_value.weight'].T.shape
        local_dim = local_dim // num_splits
        head_num = self.head_num
        size_per_head = hidden_dim // head_num
        dtype = self.dtype
        if self.int8_mode != 0:
            dtype = 'int8'
        if dtype == 'int4':
            size_per_head *= 2
        head_num = head_num // tensor_model_parallel_size
        #print(layer_num)
        w.extend([module['transformer'][f'layers.{i}.input_layernorm.weight'].to(torch.float16) for i in range(layer_num-1)])
        w.extend([module['transformer'][f'layers.{i}.input_layernorm.bias'].to(torch.float16) for i in range(layer_num-1)])
        #print(len(w))

        def add_quant_weight(tmp_weights):
            for i in tmp_weights:
                # all weights are saved in transpose
                a, b = quant_weight(i.T, dtype)
                # but in int8 mode, no need for transpose
                weight.append(a)
                scale.append(b)
                # weight.append(torch.zeros_like(a))
                # scale.append(torch.zeros_like(b))
                # from IPython import embed
                # embed()
                # print((i.T -a * b[:, None]).abs().max())

        tmp = [module['transformer'][f'layers.{i}.attention.query_key_value.weight'].reshape(3*local_dim, hidden_dim).T.to(torch.float16) for i in range(layer_num-1)]
        if dtype in ['int8', 'int4']:
            add_quant_weight(tmp)
        else:
            weight.extend(tmp)
            

        local_dim = module['transformer'][f'layers.0.attention.query_key_value.bias'].shape[0] // num_splits
        head_num = self.head_num // tensor_model_parallel_size
        size_per_head = local_dim // head_num
        w.extend([module['transformer'][f'layers.{i}.attention.query_key_value.bias'].reshape(3, local_dim).to(torch.float16) for i in range(layer_num-1)])

        tmp = [module['transformer'][f'layers.{i}.attention.dense.weight'].T.to(torch.float16) for i in range(layer_num-1)]
        if dtype in ['int8', 'int4']:
            add_quant_weight(tmp)
        else:
            weight.extend(tmp)
        
        w.extend([module['transformer'][f'layers.{i}.attention.dense.bias'].to(torch.float16) for i in range(layer_num-1)])

        w.extend([module['transformer'][f'layers.{i}.post_attention_layernorm.weight'].to(torch.float16) for i in range(layer_num-1)])
        w.extend([module['transformer'][f'layers.{i}.post_attention_layernorm.bias'].to(torch.float16) for i in range(layer_num-1)])

        local_dim = int(module['transformer']['layers.0.mlp.dense_h_to_4h.weight'].shape[0] )
        
        tmp = [module['transformer'][f'layers.{i}.mlp.dense_h_to_4h.weight'].T.to(torch.float16) for i in range(layer_num-1)]
        # from IPython import embed
        # embed()
        # exit(-1)
        if dtype in ['int8', 'int4']:
            add_quant_weight(tmp)
        else:
            weight.extend(tmp)
        
        w.extend([module['transformer'][f'layers.{i}.mlp.dense_h_to_4h.bias'].to(torch.float16) for i in range(layer_num-1)])
        
        tmp = [module['transformer'][f'layers.{i}.mlp.dense_4h_to_h.weight'].T.to(torch.float16) for i in range(layer_num-1)]
        if dtype in ['int8', 'int4']:
            add_quant_weight(tmp)
        else:
            weight.extend(tmp)

        w.extend([module['transformer'][f'layers.{i}.mlp.dense_4h_to_h.bias'].to(torch.float16) for i in range(layer_num-1)])
        #top query layer
        hidden_dim, local_dim = module['transformer']['layers.0.attention.query_key_value.weight'].T.shape
        local_dim = local_dim // num_splits
        w.extend([module['transformer'][f'topQueryLayer.input_layernorm.weight'].to(torch.float16) ])
        w.extend([module['transformer'][f'topQueryLayer.input_layernorm.bias'].to(torch.float16) ])
        
        tmp = [module['transformer'][f'topQueryLayer.attention.query.weight'].reshape(local_dim, hidden_dim).T.to(torch.float16) ]
        if dtype in ['int8', 'int4']:
            add_quant_weight(tmp)
        else:
            weight.extend(tmp)
        w.extend([module['transformer'][f'topQueryLayer.attention.query.bias'].reshape(1, local_dim).to(torch.float16) ])
        
        tmp = [module['transformer'][f'topQueryLayer.attention.key_value.weight'].reshape(2*local_dim, hidden_dim).T.to(torch.float16)]
        if dtype in ['int8', 'int4']:
            add_quant_weight(tmp)
        else:
            weight.extend(tmp)
        w.extend([module['transformer'][f'topQueryLayer.attention.key_value.bias'].reshape(2, local_dim).to(torch.float16) ])
        
        tmp = [module['transformer'][f'topQueryLayer.attention.dense.weight'].T.to(torch.float16) ]
        if dtype in ['int8', 'int4']:
            add_quant_weight(tmp)
        else:
            weight.extend(tmp)
        w.extend([module['transformer'][f'topQueryLayer.attention.dense.bias'].to(torch.float16) ])
        w.extend([module['transformer'][f'topQueryLayer.post_attention_layernorm.weight'].to(torch.float16) ])
        w.extend([module['transformer'][f'topQueryLayer.post_attention_layernorm.bias'].to(torch.float16) ])
        
        tmp = [module['transformer'][f'topQueryLayer.mlp.dense_h_to_4h.weight'].T.to(torch.float16) ]
        if dtype in ['int8', 'int4']:
            add_quant_weight(tmp)
        else:
            weight.extend(tmp)
        w.extend([module['transformer'][f'topQueryLayer.mlp.dense_h_to_4h.bias'].to(torch.float16) ])
        
        tmp = [module['transformer'][f'topQueryLayer.mlp.dense_4h_to_h.weight'].T.to(torch.float16) ]
        if dtype in ['int8', 'int4']:
            add_quant_weight(tmp)
        else:
            weight.extend(tmp)
        w.extend([module['transformer'][f'topQueryLayer.mlp.dense_4h_to_h.bias'].to(torch.float16) ])

        w.append(module[f'transformer']['final_layernorm.weight'].to(torch.float16))
        w.append(module[f'transformer']['final_layernorm.bias'].to(torch.float16))
        w.append(module[f'embedding']['position_embeddings']['weight'].to(torch.float16))
        w.append(module[f'embedding']['word_embeddings']['weight'].to(torch.float16))
        w.append(module[f'embedding']['word_embeddings']['weight'].to(torch.float16))

        w.append(module[f'topQueryEmbedding']['top_query_embeddings']['weight'].to(torch.float16))

        # Reshape
        def w_reshape(w,self_w):
            for i in range(len(w)):
                if w[i].nelement() > 0:
                    try:
                        self_w[i] = w[i].reshape(self_w[i].shape)
                    except:
                        raise RuntimeError("shape error")

        w_reshape(w, self.w)
        w_reshape(weight, self.weight)

        if self.dtype in ['int8', 'int4']:
            w_reshape(scale, self.scale)
        return True

class CODEGEEX(nn.Module):
    def __init__(self,
                 head_num, size_per_head,
                 vocab_size, start_id, end_id, layer_num,
                 max_seq_len,
                 tensor_para_size, pipeline_para_size,
                 lib_path,
                 dtype,
                 int8_mode=0):
        super().__init__()
        self.head_num = head_num
        self.size_per_head = size_per_head
        self.vocab_size = vocab_size
        self.start_id = start_id
        self.end_id = end_id
        self.layer_num = layer_num
        self.tensor_para_size = tensor_para_size
        self.pipeline_para_size = pipeline_para_size
        self.use_sparse_gemm = False
        self.build_model = False
        self.int8_mode = int8_mode

        self.dtype = dtype
        self.dtype_id = {"fp32": 0, "fp16": 1, "int8": 2, "int4": 3}[dtype]

        assert dtype in ['fp16','int8','int4'], 'unsupport data_type'

        assert torch.cuda.is_available(), "CUDA is required for this model."

        assert head_num % tensor_para_size == 0, "head_num must be a multiple of tensor_para_size."
        assert layer_num % pipeline_para_size == 0, "layer_num must be a multiple of pipeline_para_size."

        # Load the C++ model into Pytorch model.
        torch.classes.load_library(os.path.abspath(lib_path))

        # Prepare weights
        self.weights = CODEGEEXWeights(head_num, size_per_head, layer_num, vocab_size,
                                  max_seq_len, tensor_para_size, pipeline_para_size, dtype,
                                  int8_mode)

        # Prepare for tensor/pipeline parallel
        try:
            dist.init_process_group(backend='mpi')
        except:
            print("[INFO] WARNING: Have initalize the process group")
        self.rank = dist.get_rank()
        self.device_count = torch.cuda.device_count()
        self.device = self.rank % self.device_count
        torch.cuda.set_device(self.device)

        world_size = dist.get_world_size()
        assert world_size == tensor_para_size * pipeline_para_size, "tensor_para_size * pipeline_para_size must be equal to world_size."

        self.tensor_para_rank = self.rank % self.tensor_para_size
        self.pipeline_para_rank = self.rank // self.tensor_para_size

        # Create and copy model to the device.
        #self.cuda()

    def load(self, ckpt_path):
        is_load = self.weights.load(ckpt_path, tensor_para_rank=self.tensor_para_rank,
                                    pipeline_para_rank=self.pipeline_para_rank)
        self.cuda()
        return is_load

    def half(self):
        self.weights._map(lambda w: w.half())
        self.cuda()

    def bfloat16(self):
        self.weights._map(lambda w: w.bfloat16())
        self.cuda()

    def sparse(self):
        if not self.use_sparse_gemm:
            self.use_sparse_gemm = True
            self.cuda()

    def cuda(self):
        self.weights._map(lambda w: w.contiguous().cuda(self.device))
        if self.int8_mode != 0:
            self.weights._map_int8(lambda w: w.cuda(self.device))

        if self.build_model:
            del self.model
            self.build_model = False
        #print("before Op")

        self.model = torch.classes.FasterTransformer.CodegeexOp(self.head_num, self.size_per_head, 4 * self.head_num * self.size_per_head,
                                                           self.layer_num, self.vocab_size, self.start_id, self.end_id,
                                                           self.use_sparse_gemm, self.dtype_id, self.weights.w, self.weights.weight, self.weights.scale)
        #print("after Op")
        self.build_model = True

    def forward(self,
                start_ids,
                start_lengths,
                output_len,
                beam_width=1,
                top_k=1.0,
                top_p=0.0,
                beam_search_diversity_rate=0.0,
                temperature=1.0,
                len_penalty=1.0,
                repetition_penalty=1.0,
                random_seed=0,
                return_output_length=False,
                return_cum_log_probs=0):
        if not self.build_model:
            self.cuda()
        input_len = start_ids.size(1)
        assert input_len > 0, "input len must be larger than zero. For an unconditional case, use start_id as the first token."

        # Inputs to device
        start_ids = start_ids.cuda(self.device)
        start_lengths = start_lengths.cuda(self.device)
        # outputs: output_ids, output_lengths, output_cum_log_probs (optional)
        outputs = self.model.forward(start_ids,
                                     start_lengths,
                                     output_len,
                                     beam_width,
                                     top_k,
                                     top_p,
                                     beam_search_diversity_rate,
                                     temperature,
                                     len_penalty,
                                     repetition_penalty,
                                     random_seed,
                                     return_cum_log_probs)
        if return_cum_log_probs == 0:
            output_ids, output_lengths = outputs
        else:
            output_ids, output_lengths, output_cum_log_probs = outputs
        if return_output_length:
            if return_cum_log_probs > 0:
                return output_ids, output_lengths, output_cum_log_probs
            else:
                return output_ids, output_lengths
        else:
            return output_ids

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor
