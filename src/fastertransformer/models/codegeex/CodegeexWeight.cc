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

#include "src/fastertransformer/models/codegeex/CodegeexWeight.h"

namespace fastertransformer {

template<typename T>
CodegeexWeight<T>::CodegeexWeight(
    const int hidden_units, const int inter_size, const int vocab_size, const int num_layer, const int max_seq_len):
    hidden_units_(hidden_units),
    inter_size_(inter_size),
    vocab_size_(vocab_size),
    num_layer_(num_layer),
    max_seq_len_(max_seq_len)
{
    for (int l = 0; l < 1; l++) {
        topquery_layer_weights.push_back(CodegeexTopQueryLayerWeight<T>(hidden_units_, inter_size_));
    }
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(CodegeexDecoderLayerWeight<T>(hidden_units_, inter_size_));
    }

    mallocWeights();
    setWeightPtr();
}

template<typename T>
CodegeexWeight<T>::~CodegeexWeight()
{
    if (is_maintain_buffer == true) {
        for (int i = 0; i < 6; i++) {
            deviceFree(weights_ptr[i]);
        }

        topquery_position_encoding_table = nullptr;
        position_encoding_table = nullptr;
        pre_decoder_embedding_table = nullptr;
        post_decoder_layernorm.beta = nullptr;
        post_decoder_layernorm.gamma = nullptr;
        post_decoder_embedding.kernel = nullptr;
        post_decoder_embedding.bias = nullptr;
        is_maintain_buffer = false;
    }
}

template<typename T>
CodegeexWeight<T>::CodegeexWeight(const CodegeexWeight& other):
    hidden_units_(other.hidden_units_),
    inter_size_(other.inter_size_),
    vocab_size_(other.vocab_size_),
    num_layer_(other.num_layer_),
    max_seq_len_(other.max_seq_len_)
{
    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], max_seq_len_ * vocab_size_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], vocab_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * vocab_size_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], max_seq_len_ * vocab_size_);
    setWeightPtr();
    
    topquery_layer_weights.clear();
    for (int l = 0; l < 1; l++) {
        topquery_layer_weights.push_back(other.topquery_layer_weights[l]);
    }
    decoder_layer_weights.clear();
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
    }
}

template<typename T>
CodegeexWeight<T>& CodegeexWeight<T>::operator=(const CodegeexWeight& other)
{
    hidden_units_ = other.hidden_units_;
    inter_size_ = other.inter_size_;
    num_layer_ = other.num_layer_;
    vocab_size_ = other.vocab_size_;
    max_seq_len_ = other.max_seq_len_;

    mallocWeights();
    cudaD2Dcpy(weights_ptr[0], other.weights_ptr[0], max_seq_len_ * vocab_size_);
    cudaD2Dcpy(weights_ptr[1], other.weights_ptr[1], vocab_size_ * hidden_units_);
    cudaD2Dcpy(weights_ptr[2], other.weights_ptr[2], hidden_units_);
    cudaD2Dcpy(weights_ptr[3], other.weights_ptr[3], hidden_units_);
    cudaD2Dcpy(weights_ptr[4], other.weights_ptr[4], hidden_units_ * vocab_size_);
    cudaD2Dcpy(weights_ptr[5], other.weights_ptr[5], max_seq_len_ * vocab_size_);
    setWeightPtr();

    topquery_layer_weights.clear();
    for (int l = 0; l < 1; l++) {
        topquery_layer_weights.push_back(other.topquery_layer_weights[l]);
    }
    decoder_layer_weights.clear();
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights.push_back(other.decoder_layer_weights[l]);
    }
    return *this;
}

template<typename T>
void CodegeexWeight<T>::setWeightPtr()
{
    position_encoding_table = weights_ptr[0];
    pre_decoder_embedding_table = weights_ptr[1];
    post_decoder_layernorm.beta = weights_ptr[2];
    post_decoder_layernorm.gamma = weights_ptr[3];
    post_decoder_embedding.kernel = weights_ptr[4];
    post_decoder_embedding.bias = nullptr;
    topquery_position_encoding_table = weights_ptr[5];
}

template<typename T>
void CodegeexWeight<T>::mallocWeights()
{
    deviceMalloc(&weights_ptr[0], max_seq_len_ * vocab_size_);
    deviceMalloc(&weights_ptr[1], vocab_size_ * hidden_units_);
    deviceMalloc(&weights_ptr[2], hidden_units_);
    deviceMalloc(&weights_ptr[3], hidden_units_);
    deviceMalloc(&weights_ptr[4], hidden_units_ * vocab_size_);
    deviceMalloc(&weights_ptr[5], max_seq_len_ * vocab_size_);
    is_maintain_buffer = true;
}

template<typename T>
void CodegeexWeight<T>::loadModel(std::string dir_path)
{
    FT_CHECK(is_maintain_buffer == true);

    loadWeightFromBin<T>(weights_ptr[0], {max_seq_len_, hidden_units_}, dir_path + "/model.wpe.bin");
    loadWeightFromBin<T>(weights_ptr[1], {vocab_size_ * hidden_units_}, dir_path + "/model.wte.bin");
    loadWeightFromBin<T>(weights_ptr[2], {hidden_units_}, dir_path + "/model.final_layernorm.bias.bin");
    loadWeightFromBin<T>(weights_ptr[3], {hidden_units_}, dir_path + "/model.final_layernorm.weight.bin");
    loadWeightFromBin<T>(weights_ptr[4], {vocab_size_ * hidden_units_}, dir_path + "/model.wte.bin");
    loadWeightFromBin<T>(weights_ptr[5], {max_seq_len_, hidden_units_}, dir_path + "/model.topwpe.bin");

    for (int l = 0; l < 1; l++) {
        topquery_layer_weights[l].loadModel(dir_path + "/model.toplayers." + std::to_string(l));
    }
    for (int l = 0; l < num_layer_; l++) {
        decoder_layer_weights[l].loadModel(dir_path + "/model.layers." + std::to_string(l));
    }
}

#ifdef SPARSITY_ENABLED
template<typename T>
void CodegeexWeight<T>::compress_weights(cublasMMWrapper& cublas_wrapper)
{
    // Assertion to prevent invalid attributes. By now, codegeex_weight may not
    // have proper attribute values, because one can directly modify decoder
    // layer weights from outside.
    FT_CHECK(decoder_layer_weights.size() == static_cast<size_t>(num_layer_-1));
    for (int i = 0; i < 1; ++i) {
        topquery_layer_weights[i].compress_weights(cublas_wrapper, hidden_units_);
    }
    for (int i = 0; i < num_layer_; ++i) {
        decoder_layer_weights[i].compress_weights(cublas_wrapper, hidden_units_);
    }
}
#endif

template struct CodegeexWeight<float>;
template struct CodegeexWeight<half>;

}  // namespace fastertransformer
