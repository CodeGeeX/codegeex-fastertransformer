/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "src/fastertransformer/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "cutlass/numeric_types.h"

#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/kernels/matrix_vector_multiplication.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <vector>

namespace fastertransformer {


template<typename T>
class FfnLayer: public BaseLayer {
private:
    // buffer handling
    size_t max_token_num_ = 0;

    // meta data
    size_t head_num_;
    size_t size_per_head_;

    // int8_mode_ == 1 for weight quantized only gemm for GPT
    int int8_mode_ = 0;

    // calculated data
    size_t hidden_units_;

    std::shared_ptr<CutlassFpAIntBGemmRunner<T, uint8_t>> weight_only_int8_fc_runner_;
    std::shared_ptr<CutlassFpAIntBGemmRunner<T, cutlass::uint4b_t>> weight_only_int4_fc_runner_;

    void allocateBuffer() override;
    void freeBuffer() override;
    bool isValidTokenNum(size_t token_num);
    void allocateBuffer(size_t token_num);

protected:
    T* inter_buf_ = nullptr;
    T* weights_buf_ = nullptr;

    char*  mixed_gemm_workspace_ = nullptr;
    size_t mixed_gemm_ws_bytes_  = 0;

    size_t inter_size_;
    virtual void invokeAddBiasActivation(const int m, const T* bias) = 0;

public:
    FfnLayer(size_t max_batch_size,
             size_t max_seq_len,
             size_t head_num,
             size_t size_per_head,
             size_t inter_size,
             cudaStream_t stream,
             cublasMMWrapper* cublas_wrapper,
             IAllocator* allocator,
             bool is_free_buffer_after_forward,
             bool sparse = false,
             int int8_mode = 0);

    FfnLayer(FfnLayer<T> const& ffn_layer);

    virtual ~FfnLayer();

    virtual void forward(std::vector<fastertransformer::Tensor>* output_tensors,
                         const std::vector<fastertransformer::Tensor>* input_tensors,
                         const FfnWeight<T>* ffn_weights);
};

template<typename T>
class GeluFfnLayer: public FfnLayer<T> {
public:
    GeluFfnLayer(size_t max_batch_size,
                 size_t max_seq_len,
                 size_t head_num,
                 size_t size_per_head,
                 size_t inter_size,
                 cudaStream_t stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator* allocator,
                 bool is_free_buffer_after_forward,
                 bool sparse = false,
                 int int8_mode = 0);

    GeluFfnLayer(GeluFfnLayer<T> const& ffn_layer);

    virtual ~GeluFfnLayer() = default;

protected:
    using FfnLayer<T>::stream_;

private:
    using FfnLayer<T>::inter_buf_;
    using FfnLayer<T>::inter_size_;
    void invokeAddBiasActivation(const int m, const T* bias) override;
};

template<typename T>
class FastGeluFfnLayer: public FfnLayer<T> {
public:
    FastGeluFfnLayer(size_t max_batch_size,
                 size_t max_seq_len,
                 size_t head_num,
                 size_t size_per_head,
                 size_t inter_size,
                 cudaStream_t stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator* allocator,
                 bool is_free_buffer_after_forward,
                 bool sparse = false,
                 int int8_mode = 0);

    FastGeluFfnLayer(FastGeluFfnLayer<T> const& ffn_layer);

    virtual ~FastGeluFfnLayer() = default;

protected:
    using FfnLayer<T>::stream_;

private:
    using FfnLayer<T>::inter_buf_;
    using FfnLayer<T>::inter_size_;
    void invokeAddBiasActivation(const int m, const T* bias) override;
};

template<typename T>
class ReluFfnLayer: public FfnLayer<T> {
public:
    ReluFfnLayer(size_t max_batch_size,
                 size_t max_seq_len,
                 size_t head_num,
                 size_t size_per_head,
                 size_t inter_size,
                 cudaStream_t stream,
                 cublasMMWrapper* cublas_wrapper,
                 IAllocator* allocator,
                 bool is_free_buffer_after_forward,
                 bool sparse = false);

    ReluFfnLayer(ReluFfnLayer<T> const& ffn_layer);

    virtual ~ReluFfnLayer() = default;

protected:
    using FfnLayer<T>::stream_;

private:
    using FfnLayer<T>::inter_buf_;
    using FfnLayer<T>::inter_size_;
    void invokeAddBiasActivation(const int m, const T* bias) override;
};

}  // namespace fastertransformer
