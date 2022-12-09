# Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

cmake_minimum_required(VERSION 3.8)

add_library(ParallelCodegeexDecoderLayerWeight STATIC ParallelCodegeexDecoderLayerWeight.cc)
set_property(TARGET ParallelCodegeexDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelCodegeexDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelCodegeexDecoderLayerWeight PUBLIC memory_utils calibrate_quantize_weight_kernels)

add_library(ParallelCodegeexTopQueryLayerWeight STATIC ParallelCodegeexTopQueryLayerWeight.cc)
set_property(TARGET ParallelCodegeexTopQueryLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelCodegeexTopQueryLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelCodegeexTopQueryLayerWeight PUBLIC memory_utils calibrate_quantize_weight_kernels)

add_library(ParallelCodegeexWeight STATIC ParallelCodegeexWeight.cc)
set_property(TARGET ParallelCodegeexWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelCodegeexWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelCodegeexWeight PUBLIC ParallelCodegeexDecoderLayerWeight ParallelCodegeexTopQueryLayerWeight)

add_library(ParallelCodegeexContextDecoder STATIC ParallelCodegeexContextDecoder.cc)
set_property(TARGET ParallelCodegeexContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelCodegeexContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelCodegeexContextDecoder PUBLIC -lcudart TensorParallelFastGeluFfnLayer
	TensorParallelGptContextAttentionLayer TensorParallelCodegeexTopContextAttentionLayer layernorm_kernels
                                                add_residual_kernels nccl_utils codegeex_kernels)

add_library(ParallelCodegeexDecoder STATIC ParallelCodegeexDecoder.cc)
set_property(TARGET ParallelCodegeexDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelCodegeexDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelCodegeexDecoder PUBLIC -lcudart TensorParallelFastGeluFfnLayer
	TensorParallelDecoderSelfAttentionLayer TensorParallelDecoderTopSelfAttentionLayer layernorm_kernels
                                                add_residual_kernels nccl_utils)

add_library(ParallelCodegeex STATIC ParallelCodegeex.cc)
set_property(TARGET ParallelCodegeex PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelCodegeex PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelCodegeex PUBLIC -lcudart ParallelCodegeexDecoder ParallelCodegeexContextDecoder decoding_kernels codegeex_kernels
                      DynamicDecodeLayer BaseBeamSearchLayer bert_preprocess_kernels ParallelCodegeexWeight custom_ar_comm logprob_kernels)

add_executable(codegeex_gemm codegeex_gemm.cc)
target_link_libraries(codegeex_gemm PUBLIC -lcudart codegeex_gemm_func memory_utils)
