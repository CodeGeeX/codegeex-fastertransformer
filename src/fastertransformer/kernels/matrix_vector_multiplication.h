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

#include "int8_utils.cuh"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastertransformer {

#define MaxPerChannelLdkMultiplicationNum 8

template<typename T>
void int8WeightPerChannelLdkMultiplicationLauncher(const int8_t* weight,
                                                   const T* input,
                                                   const T* scale_list,
                                                   T* output,
                                                   const int M_SIZE,
                                                   const int n,
                                                   const int k,
                                                   cudaStream_t stream);


template<typename T>
void int4WeightPerChannelLdkMultiplicationLauncher(const int8_t* weight,
                                                   const T* input,
                                                   const T* scale_list,
                                                   T* output,
                                                   const int M_SIZE,
                                                   const int n,
                                                   const int k,
                                                   cudaStream_t stream);

template<typename T>
void invokeInt4WeightExtractionTrans(const int8_t* weight,
                                const T* scale_list,
                                T* output,
                                const int n,
                                const int k,
                                cudaStream_t stream);


template<typename T>
void invokeInt4WeightExtractionNoTrans(const int8_t* weight,
                                const T* scale_list,
                                T* output,
                                const int n,
                                const int k,
                                cudaStream_t stream);

template<typename T>
void invokeInt8WeightExtractionTrans(const int8_t* weight,
                                const T* scale_list,
                                T* output,
                                const int n,
                                const int k,
                                cudaStream_t stream);

template<typename T>
void invokeInt8WeightExtractionNoTrans(const int8_t* weight,
                                const T* scale_list,
                                T* output,
                                const int n,
                                const int k,
                                cudaStream_t stream);

#define invokeMixWeightGemm(DENSE_WEIGHT,WEIGHT_BUF,INPUT_TENSOR,OUTPUT_TENSOR,N_SIZE,M_SIZE,K_SIZE,CUBLAS_WRAPPER,STREAM) \
    if (DENSE_WEIGHT.kernel) { \
        CUBLAS_WRAPPER->Gemm(CUBLAS_OP_N, \
                        CUBLAS_OP_N, \
                        N_SIZE, \
                        M_SIZE, \
                        K_SIZE, \
                        DENSE_WEIGHT.kernel, \
                        N_SIZE, \
                        INPUT_TENSOR, \
                        K_SIZE, \
                        OUTPUT_TENSOR, \
                        N_SIZE); \
    } else if(M_SIZE > MaxPerChannelLdkMultiplicationNum) { \
        FT_CHECK(DENSE_WEIGHT.int8_kernel != nullptr || DENSE_WEIGHT.int4_kernel != nullptr); \
        FT_CHECK(DENSE_WEIGHT.quant_scale != nullptr); \
        if(DENSE_WEIGHT.int8_kernel != nullptr) { \
            invokeInt8WeightExtractionNoTrans(DENSE_WEIGHT.int8_kernel, \
                                    DENSE_WEIGHT.quant_scale, \
                                    WEIGHT_BUF, \
                                    N_SIZE, \
                                    K_SIZE, \
                                    STREAM); \
        } else { \
            invokeInt4WeightExtractionNoTrans(DENSE_WEIGHT.int4_kernel, \
                                    DENSE_WEIGHT.quant_scale, \
                                    WEIGHT_BUF, \
                                    N_SIZE, \
                                    K_SIZE, \
                                    STREAM); \
        } \
        sync_check_cuda_error(); \
        CUBLAS_WRAPPER->Gemm(CUBLAS_OP_T, \
                        CUBLAS_OP_N, \
                        N_SIZE, \
                        M_SIZE, \
                        K_SIZE, \
                        WEIGHT_BUF, \
                        K_SIZE, \
                        INPUT_TENSOR, \
                        K_SIZE, \
                        OUTPUT_TENSOR, \
                        N_SIZE); \
    } else { \
        FT_CHECK(DENSE_WEIGHT.int8_kernel != NULL || DENSE_WEIGHT.int4_kernel != NULL); \
        FT_CHECK(DENSE_WEIGHT.quant_scale != NULL); \
        if(DENSE_WEIGHT.int8_kernel != NULL) { \
            int8WeightPerChannelLdkMultiplicationLauncher(DENSE_WEIGHT.int8_kernel, \
                                                    INPUT_TENSOR, \
                                                    DENSE_WEIGHT.quant_scale, \
                                                    OUTPUT_TENSOR, \
                                                    M_SIZE, \
                                                    N_SIZE, \
                                                    K_SIZE, \
                                                    STREAM); \
        } else { \
            int4WeightPerChannelLdkMultiplicationLauncher(DENSE_WEIGHT.int4_kernel, \
                                                    INPUT_TENSOR, \
                                                    DENSE_WEIGHT.quant_scale, \
                                                    OUTPUT_TENSOR, \
                                                    M_SIZE, \
                                                    N_SIZE, \
                                                    K_SIZE, \
                                                    STREAM); \
        } \
    } \
    sync_check_cuda_error();


}  // namespace fastertransformer
