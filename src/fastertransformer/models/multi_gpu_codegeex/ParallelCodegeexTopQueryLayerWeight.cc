/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/models/multi_gpu_codegeex/ParallelCodegeexTopQueryLayerWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"

namespace fastertransformer {

template<typename T>
ParallelCodegeexTopQueryLayerWeight<T>::ParallelCodegeexTopQueryLayerWeight(const int hidden_units,
                                                                const int inter_size,
                                                                const int tensor_para_size,
                                                                const int tensor_para_rank,
                                                                const int int8_mode):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    tensor_para_size_(tensor_para_size),
    tensor_para_rank_(tensor_para_rank),
    int8_mode_(int8_mode)
{
    mallocWeights();
    setWeightPtr();
}

template<typename T>
ParallelCodegeexTopQueryLayerWeight<T>::ParallelCodegeexTopQueryLayerWeight(const int int8_mode): int8_mode_(int8_mode)
{
}

template<typename T>
ParallelCodegeexTopQueryLayerWeight<T>::~ParallelCodegeexTopQueryLayerWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < 9; i++) {
            deviceFree(weights_ptr[i]);
        }

        for (int i = 0; i < 5; i++) {
            deviceFree(kernel_ptr[i]);
        }
        pre_layernorm_weights.beta = nullptr;
        pre_layernorm_weights.gamma = nullptr;
        self_attention_weights.query_weight.kernel = nullptr;
        self_attention_weights.query_weight.bias = nullptr;
        self_attention_weights.key_weight.kernel = nullptr;
        self_attention_weights.key_weight.bias = nullptr;
        self_attention_weights.attention_output_weight.kernel = nullptr;
        self_attention_weights.attention_output_weight.bias = nullptr;
        self_attn_layernorm_weights.beta = nullptr;
        self_attn_layernorm_weights.gamma = nullptr;

        ffn_weights.intermediate_weight.kernel = nullptr;
        ffn_weights.intermediate_weight.bias = nullptr;
        ffn_weights.output_weight.kernel = nullptr;
        ffn_weights.output_weight.bias = nullptr;

        if (int8_mode_ != 0) {
            for (int i = 0; i < 5; i++) {
                deviceFree(int8_weights_ptr[i]);
                deviceFree(scale_ptr[i]);
            }
            self_attention_weights.query_weight.int8_kernel = nullptr;
            self_attention_weights.query_weight.quant_scale = nullptr;
            self_attention_weights.key_weight.int8_kernel = nullptr;
            self_attention_weights.key_weight.quant_scale = nullptr;
            self_attention_weights.attention_output_weight.int8_kernel = nullptr;
            self_attention_weights.attention_output_weight.quant_scale = nullptr;
            ffn_weights.intermediate_weight.int8_kernel = nullptr;
            ffn_weights.intermediate_weight.quant_scale = nullptr;
            ffn_weights.output_weight.int8_kernel = nullptr;
            ffn_weights.output_weight.quant_scale = nullptr;
        }

        is_maintain_buffer = false;
    }
}

template<typename T>
ParallelCodegeexTopQueryLayerWeight<T>::ParallelCodegeexTopQueryLayerWeight(const ParallelCodegeexTopQueryLayerWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    tensor_para_size_(other.tensor_para_size_),
    tensor_para_rank_(other.tensor_para_rank_),
    int8_mode_(other.int8_mode_)
{
    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], 1 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 2 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);

    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_);

    if (int8_mode_ == 0) {
        cudaD2Dcpy(kernel_ptr[0], other.kernel_ptr[0], hidden_units_ * 1 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(kernel_ptr[1], other.kernel_ptr[1], hidden_units_ * 2 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(kernel_ptr[2], other.kernel_ptr[2], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(kernel_ptr[3], other.kernel_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(kernel_ptr[4], other.kernel_ptr[4], inter_size_ / tensor_para_size_ * hidden_units_);
    }
    else if (int8_mode_ == 1) {
        cudaD2Dcpy(
            int8_weights_ptr[0], other.int8_weights_ptr[0], hidden_units_ * 1 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(
            int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ * 2 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[4], other.int8_weights_ptr[4], inter_size_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(scale_ptr[0], other.scale_ptr[0], 1 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[1], other.scale_ptr[1], 2 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[2], other.scale_ptr[2], hidden_units_);
        cudaD2Dcpy(scale_ptr[3], other.scale_ptr[3], inter_size_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[4], other.scale_ptr[4], hidden_units_);
    }

    setWeightPtr();
}

template<typename T>
ParallelCodegeexTopQueryLayerWeight<T>&
ParallelCodegeexTopQueryLayerWeight<T>::operator=(const ParallelCodegeexTopQueryLayerWeight& other)
{
    hidden_units_ = other.hidden_units_;
    inter_size_ = other.inter_size_;
    tensor_para_size_ = other.tensor_para_size_;
    tensor_para_rank_ = other.tensor_para_rank_;
    int8_mode_ = other.int8_mode_;

    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], hidden_units_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], 1 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], 2 * hidden_units_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], hidden_units_);
    cudaD2Dcpy(weights_ptr[6], other.weights_ptr[6], hidden_units_);

    cudaD2Dcpy(weights_ptr[7], other.weights_ptr[7], inter_size_ / tensor_para_size_);
    cudaD2Dcpy(weights_ptr[8], other.weights_ptr[8], hidden_units_);

    if (int8_mode_ == 0) {
        cudaD2Dcpy(kernel_ptr[0], other.kernel_ptr[0], hidden_units_ * 1 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(kernel_ptr[1], other.kernel_ptr[1], hidden_units_ * 2 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(kernel_ptr[2], other.kernel_ptr[2], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(kernel_ptr[3], other.kernel_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(kernel_ptr[4], other.kernel_ptr[4], inter_size_ / tensor_para_size_ * hidden_units_);
    }
    else if (int8_mode_ == 1) {
        cudaD2Dcpy(
            int8_weights_ptr[0], other.int8_weights_ptr[0], hidden_units_ * 1 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(
            int8_weights_ptr[1], other.int8_weights_ptr[1], hidden_units_ * 2 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[2], other.int8_weights_ptr[2], hidden_units_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(int8_weights_ptr[3], other.int8_weights_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_);
        cudaD2Dcpy(int8_weights_ptr[4], other.int8_weights_ptr[4], inter_size_ / tensor_para_size_ * hidden_units_);
        cudaD2Dcpy(scale_ptr[0], other.scale_ptr[0], 1 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[1], other.scale_ptr[1], 2 * hidden_units_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[2], other.scale_ptr[2], hidden_units_);
        cudaD2Dcpy(scale_ptr[3], other.scale_ptr[3], inter_size_ / tensor_para_size_);
        cudaD2Dcpy(scale_ptr[4], other.scale_ptr[4], hidden_units_);
    }

    setWeightPtr();
    return *this;
}

template<typename T>
void ParallelCodegeexTopQueryLayerWeight<T>::loadModel(std::string dir_path, FtCudaDataType model_file_type)
{
    FT_CHECK(is_maintain_buffer == true);
    if (int8_mode_ == 0) {
    loadWeightFromBin<T>(weights_ptr[0], {(int)hidden_units_}, dir_path + ".input_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[1], {(int)hidden_units_}, dir_path + ".input_layernorm.weight.bin", model_file_type);
    loadWeightFromBin<T>(weights_ptr[2],
                         {(int)hidden_units_, (int)(1 * hidden_units_ / tensor_para_size_)},
                         dir_path + ".attention.query.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[3],
                         {1, (int)(hidden_units_ / tensor_para_size_)},
                         dir_path + ".attention.query.bias." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[4],
                         {(int)hidden_units_, (int)(2 * hidden_units_ / tensor_para_size_)},
                         dir_path + ".attention.key_value.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[5],
                         {2, (int)(hidden_units_ / tensor_para_size_)},
                         dir_path + ".attention.key_value.bias." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[6],
                         {(int)(hidden_units_ / tensor_para_size_), (int)hidden_units_},
                         dir_path + ".attention.dense.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[7], {(int)hidden_units_}, dir_path + ".attention.dense.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[8], {(int)hidden_units_}, dir_path + ".post_attention_layernorm.bias.bin", model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[9], {(int)hidden_units_}, dir_path + ".post_attention_layernorm.weight.bin", model_file_type);

    loadWeightFromBin<T>(weights_ptr[10],
                         {(int)hidden_units_, (int)(inter_size_ / tensor_para_size_)},
                         dir_path + ".mlp.dense_h_to_4h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[11],
                         {(int)(inter_size_ / tensor_para_size_)},
                         dir_path + ".mlp.dense_h_to_4h.bias." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(weights_ptr[12],
                         {(int)(inter_size_ / tensor_para_size_), (int)hidden_units_},
                         dir_path + ".mlp.dense_4h_to_h.weight." + std::to_string(tensor_para_rank_) + ".bin",
                         model_file_type);
    loadWeightFromBin<T>(
        weights_ptr[13], {(int)hidden_units_}, dir_path + ".mlp.dense_4h_to_h.bias.bin", model_file_type);
    }
}

template<typename T>
void ParallelCodegeexTopQueryLayerWeight<T>::setWeightPtr()
{
    pre_layernorm_weights.beta = weights_ptr[0];
    pre_layernorm_weights.gamma = weights_ptr[1];
    self_attention_weights.query_weight.bias = weights_ptr[2];
    self_attention_weights.key_weight.bias = weights_ptr[3];
    self_attention_weights.attention_output_weight.bias = weights_ptr[4];
    self_attn_layernorm_weights.beta = weights_ptr[5];
    self_attn_layernorm_weights.gamma = weights_ptr[6];

    ffn_weights.intermediate_weight.bias = weights_ptr[7];
    ffn_weights.output_weight.bias = weights_ptr[8];
    
    self_attention_weights.query_weight.kernel = nullptr;
    self_attention_weights.key_weight.kernel = nullptr;
    self_attention_weights.attention_output_weight.kernel = nullptr;
    ffn_weights.intermediate_weight.kernel = nullptr;
    ffn_weights.output_weight.kernel = nullptr;

    if (int8_mode_ == 0) {
        self_attention_weights.query_weight.kernel =
            kernel_ptr[0];  // hidden_units_ * 1 * hidden_units_ / tensor_para_size_
        self_attention_weights.key_weight.kernel =
            kernel_ptr[1];  // hidden_units_ * 2 * hidden_units_ / tensor_para_size_
        self_attention_weights.attention_output_weight.kernel =
            kernel_ptr[2];  // hidden_units_ / tensor_para_size_ * hidden_units_
        ffn_weights.intermediate_weight.kernel =
            kernel_ptr[3];  // hidden_units_ * inter_size_ / tensor_para_size_
        ffn_weights.output_weight.kernel = kernel_ptr[4];  // inter_size_ / tensor_para_size_ * hidden_units_
    }
    else {
        self_attention_weights.query_weight.int8_kernel = int8_weights_ptr[0];
        self_attention_weights.key_weight.int8_kernel = int8_weights_ptr[1];
        self_attention_weights.attention_output_weight.int8_kernel = int8_weights_ptr[2];
        ffn_weights.intermediate_weight.int8_kernel = int8_weights_ptr[3];
        ffn_weights.output_weight.int8_kernel = int8_weights_ptr[4];
        self_attention_weights.query_weight.quant_scale = scale_ptr[0];
        self_attention_weights.key_weight.quant_scale = scale_ptr[1];
        self_attention_weights.attention_output_weight.quant_scale = scale_ptr[2];
        ffn_weights.intermediate_weight.quant_scale = scale_ptr[3];
        ffn_weights.output_weight.quant_scale = scale_ptr[4];
    }

    is_maintain_buffer = true;
}

template<typename T>
void ParallelCodegeexTopQueryLayerWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], hidden_units_);
    deviceMalloc(&weights_ptr[1], hidden_units_);
    deviceMalloc(&weights_ptr[2], 1 * hidden_units_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[3], 2 * hidden_units_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[4], hidden_units_);
    deviceMalloc(&weights_ptr[5], hidden_units_);
    deviceMalloc(&weights_ptr[6], hidden_units_);

    deviceMalloc(&weights_ptr[7], inter_size_ / tensor_para_size_);
    deviceMalloc(&weights_ptr[8], hidden_units_);

    if (int8_mode_ == 0) {
        deviceMalloc(&kernel_ptr[0], hidden_units_ * 1 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&kernel_ptr[1], hidden_units_ * 2 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&kernel_ptr[2], hidden_units_ / tensor_para_size_ * hidden_units_);
        deviceMalloc(&kernel_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_);
        deviceMalloc(&kernel_ptr[4], inter_size_ / tensor_para_size_ * hidden_units_);
    }
    else if (int8_mode_ == 1) {
        deviceMalloc(&int8_weights_ptr[0], hidden_units_ * 1 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[1], hidden_units_ * 2 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[2], hidden_units_ / tensor_para_size_ * hidden_units_);
        deviceMalloc(&int8_weights_ptr[3], hidden_units_ * inter_size_ / tensor_para_size_);
        deviceMalloc(&int8_weights_ptr[4], inter_size_ / tensor_para_size_ * hidden_units_);

        deviceMalloc(&scale_ptr[0], 1 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&scale_ptr[1], 2 * hidden_units_ / tensor_para_size_);
        deviceMalloc(&scale_ptr[2], hidden_units_);
        deviceMalloc(&scale_ptr[3], inter_size_ / tensor_para_size_);
        deviceMalloc(&scale_ptr[4], hidden_units_);
    }
}

template struct ParallelCodegeexTopQueryLayerWeight<float>;
template struct ParallelCodegeexTopQueryLayerWeight<half>;
#ifdef ENABLE_BF16
template struct ParallelCodegeexTopQueryLayerWeight<__nv_bfloat16>;
#endif

}  // namespace fastertransformer
