# Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

add_library(CodegeexDecoderLayerWeight STATIC CodegeexDecoderLayerWeight.cc)
set_property(TARGET CodegeexDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET CodegeexDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(CodegeexDecoderLayerWeight PUBLIC memory_utils)

add_library(CodegeexTopQueryLayerWeight STATIC CodegeexTopQueryLayerWeight.cc)
set_property(TARGET CodegeexTopQueryLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET CodegeexTopQueryLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(CodegeexTopQueryLayerWeight PUBLIC memory_utils)

add_library(CodegeexDecoder STATIC CodegeexDecoder.cc)
set_property(TARGET CodegeexDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET CodegeexDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(CodegeexDecoder PUBLIC -lcudart cublasMMWrapper 
                                    DecoderSelfAttentionLayer DecoderTopSelfAttentionLayer FfnLayer layernorm_kernels add_residual_kernels CodegeexDecoderLayerWeight CodegeexTopQueryLayerWeight)

add_library(CodegeexContextDecoder STATIC CodegeexContextDecoder.cc)
set_property(TARGET CodegeexContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET CodegeexContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(CodegeexContextDecoder PUBLIC -lcudart cublasMMWrapper 
                        GptContextAttentionLayer CodegeexTopContextAttentionLayer FfnLayer layernorm_kernels add_residual_kernels)

add_library(CodegeexWeight STATIC CodegeexWeight.cc)
set_property(TARGET CodegeexWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET CodegeexWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(CodegeexWeight PUBLIC CodegeexDecoderLayerWeight CodegeexTopQueryLayerWeight)

add_library(Codegeex STATIC Codegeex.cc)
set_property(TARGET Codegeex PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Codegeex PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Codegeex PUBLIC -lcudart CodegeexDecoder CodegeexContextDecoder decoding_kernels codegeex_kernels
                        OnlineBeamSearchLayer BeamSearchLayer TopKSamplingLayer TopPSamplingLayer 
                        TopKTopPSamplingLayer bert_preprocess_kernels CodegeexWeight)
