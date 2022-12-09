/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

using std::vector;

class IFGpt {
public:
    virtual ~IFGpt() {}
    virtual void forward(th::Tensor& input_ids,
                         th::Tensor& input_lengths,
                         th::Tensor& output_ids,
                         th::Tensor& parent_ids,
                         th::Tensor& sequence_lengths,
                         th::Tensor& cum_log_probs,
                         const size_t request_output_len,
                         const size_t beam_width,
                         const size_t top_k,
                         const float top_p,
                         const float beam_search_diversity_rate,
                         const float temperature,
                         const float len_penalty,
                         const float repetition_penalty,
                         const unsigned long long int random_seed,
                         const int return_cum_log_probs = 0) = 0;
};

template<typename T>
class FTGpt: public IFGpt {
public:
    FTGpt(const size_t head_num,
          const size_t size_per_head,
          const size_t inter_size,
          const size_t layer_num,
          const size_t vocab_size,
          const int start_id,
          const int end_id,
          const bool sparse,
          const vector<th::Tensor> weights):
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        layer_num_(layer_num),
        vocab_size_(vocab_size),
        start_id_(start_id),
        end_id_(end_id),
#ifndef SPARSITY_ENABLED
        sparse_(false),
#else
        sparse_(sparse),
#endif
        weights_(weights)
    {
        ft::check_cuda_error(cublasLtCreate(&cublasltHandle_));
        if (sparse) {
#ifdef SPARSITY_ENABLED
            CHECK_CUSPARSE(cusparseLtInit(&cusparseLtHandle_));
#else
            std::cout << "[WARNING] Sparsity support is not enabled. Will use dense GEMM instead. "
                         "To enabled sparisty, please provide `-DSUPPORT_SPARITY` flag for compliation."
                      << std::endl;
#endif
        }

        std::string sp_config_fname = sparse ? SPGEMM_CONFIG : "";
        cublas_algo_map_ = new ft::cublasAlgoMap(GEMM_CONFIG, sp_config_fname);
        cublas_wrapper_mutex_ = new std::mutex();

        gpt_weights_.resizeLayer(layer_num_);
        for (int i = 0; i < (int)layer_num_; i++) {
            gpt_weights_.decoder_layer_weights[i]->pre_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 0 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->pre_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 1 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.kernel =
                get_ptr<T>(weights_[i + 2 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.bias =
                get_ptr<T>(weights_[i + 3 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(weights_[i + 4 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.bias =
                get_ptr<T>(weights_[i + 5 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attn_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 6 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attn_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 7 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(weights_[i + 8 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.bias =
                get_ptr<T>(weights_[i + 9 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.kernel =
                get_ptr<T>(weights_[i + 10 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.bias =
                get_ptr<T>(weights_[i + 11 * layer_num_]);
        }

        gpt_weights_.post_decoder_layernorm.gamma = get_ptr<T>(weights_[12 * layer_num_ + 0]);
        gpt_weights_.post_decoder_layernorm.beta = get_ptr<T>(weights_[12 * layer_num_ + 1]);
        gpt_weights_.position_encoding_table = get_ptr<T>(weights_[12 * layer_num_ + 2]);
        gpt_weights_.pre_decoder_embedding_table = get_ptr<T>(weights_[12 * layer_num_ + 3]);
        gpt_weights_.post_decoder_embedding.kernel = get_ptr<T>(weights_[12 * layer_num_ + 4]);
#ifdef SPARSITY_ENABLED
        if (sparse_) {
            auto stream = at::cuda::getCurrentCUDAStream().stream();
            cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
            cublasSetStream(cublas_handle, stream);
            ft::cublasMMWrapper cublas_wrapper = ft::cublasMMWrapper(cublas_handle,
                                                                     cublasltHandle_,
                                                                     cusparseLtHandle_,
                                                                     stream,
                                                                     cublas_algo_map_,
                                                                     cublas_wrapper_mutex_,
                                                                     nullptr);
            // Here we need to pass hidden_units to compress weights as sparse BERT did,
            // because GptWeights has no proper attribute value - like num_layer, dummy hidden_units,
            // or inter_size. Let me udpate an initalization of GptWeights in future.
            int hidden_units = head_num_ * size_per_head_;
            for (size_t i = 0; i < layer_num_; ++i) {
                gpt_weights_.decoder_layer_weights[i]->compress_weights(cublas_wrapper, hidden_units);
            }
            is_spmm_compressed = true;
        }
#endif

        ft::check_cuda_error(cudaGetDeviceProperties(&prop_, 0));
    }

    ~FTGpt() override
    {
        cublasLtDestroy(cublasltHandle_);
        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void forward(th::Tensor& input_ids,
                 th::Tensor& input_lengths,
                 th::Tensor& output_ids,
                 th::Tensor& parent_ids,
                 th::Tensor& sequence_lengths,
                 th::Tensor& cum_log_probs,
                 const size_t request_output_len,
                 const size_t beam_width,
                 const size_t top_k,
                 const float top_p,
                 const float beam_search_diversity_rate,
                 const float temperature,
                 const float len_penalty,
                 const float repetition_penalty,
                 const unsigned long long int random_seed,
                 const int return_cum_log_probs = 0) override
    {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublasHandle, stream);
        ft::Allocator<ft::AllocatorType::TH> allocator = ft::Allocator<ft::AllocatorType::TH>();
        ft::cublasMMWrapper cublas_wrapper = ft::cublasMMWrapper(cublasHandle,
                                                                 cublasltHandle_,
#ifdef SPARSITY_ENABLED
                                                                 cusparseLtHandle_,
#endif
                                                                 stream,
                                                                 cublas_algo_map_,
                                                                 cublas_wrapper_mutex_,
                                                                 &allocator);

        if (std::is_same<T, half>::value) {
            cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
        }
#ifdef ENABLE_BF16
        else if (std::is_same<T, __nv_bfloat16>::value) {
            cublas_wrapper.setBF16GemmConfig();
        }
#endif
        else if (std::is_same<T, float>::value) {
            cublas_wrapper.setFP32GemmConfig();
        }

        const size_t request_batch_size = (size_t)input_ids.size(0);
        const size_t max_input_length = (size_t)input_ids.size(1);
        const int total_output_len = (int)(max_input_length + request_output_len);

        ft::NcclParam tensor_para;
        ft::NcclParam pipeline_para;

        ft::ParallelGpt<T> gpt = ft::ParallelGpt<T>(request_batch_size,
                                                    total_output_len,
                                                    max_input_length,
                                                    beam_width,
                                                    head_num_,
                                                    size_per_head_,
                                                    inter_size_,
                                                    layer_num_,
                                                    vocab_size_,
                                                    start_id_,
                                                    end_id_,
                                                    beam_search_diversity_rate,
                                                    top_k,
                                                    top_p,
                                                    random_seed,
                                                    temperature,
                                                    len_penalty,
                                                    repetition_penalty,
                                                    tensor_para,
                                                    pipeline_para,
                                                    stream,
                                                    &cublas_wrapper,
                                                    &allocator,
                                                    false,
                                                    &prop_,
                                                    sparse_,
                                                    0);

        std::unordered_map<std::string, ft::Tensor> input_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"input_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, max_input_length},
                        get_ptr<int>(input_ids)}},
            {"input_lengths",
             ft::Tensor{
                 ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, get_ptr<int>(input_lengths)}},
            {"max_output_seq_len",
             ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &total_output_len}}};
        if (top_k == 0 && top_p == 0.0f) {
            ft::FT_CHECK(beam_width > 1);
            input_tensors.insert(
                {"beam_search_diversity_rate",
                 ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &beam_search_diversity_rate}});
        }
        else {
            if (top_p != 0.0f) {
                input_tensors.insert(
                    {"runtime_top_p", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &top_p}});
            }
            if (top_k != 0) {
                input_tensors.insert(
                    {"runtime_top_k", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1}, &top_k}});
            }
        }
        input_tensors.insert(
            {"temperature", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &temperature}});
        input_tensors.insert(
            {"len_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &len_penalty}});
        input_tensors.insert({"repetition_penalty",
                              ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1}, &repetition_penalty}});
        input_tensors.insert(
            {"random_seed", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT64, std::vector<size_t>{1}, &random_seed}});

        bool return_context_cum_log_probs = false;
        if (return_cum_log_probs == 2) {
            return_context_cum_log_probs = true;
            input_tensors.insert(
                {"is_return_context_cum_log_probs",
                 ft::Tensor{ft::MEMORY_CPU, ft::TYPE_BOOL, std::vector<size_t>{1}, &return_context_cum_log_probs}});
        }

        std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"output_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                        get_ptr<int>(output_ids)}},
            {"parent_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{(size_t)total_output_len, request_batch_size, beam_width},
                        get_ptr<int>(parent_ids)}},
            {"sequence_length",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width},
                        get_ptr<int>(sequence_lengths)}}};

        if (return_cum_log_probs > 0) {
            output_tensors.insert({"cum_log_probs",
                                   ft::Tensor{ft::MEMORY_GPU,
                                              ft::TYPE_FP32,
                                              std::vector<size_t>{request_batch_size, beam_width},
                                              get_ptr<float>(cum_log_probs)}});
        }

        try {
            gpt.forward(&output_tensors, &input_tensors, &gpt_weights_);
        }
        catch (std::runtime_error& error) {
            std::cout << error.what();
            exit(-1);
        }
        catch (...) {
            std::cout << "Runtime error";
            exit(-1);
        }
    }

private:
    const size_t head_num_;
    const size_t size_per_head_;
    const size_t inter_size_;
    const size_t layer_num_;
    const size_t vocab_size_;
    const int start_id_;
    const int end_id_;
    const bool sparse_;

    std::vector<th::Tensor> weights_;
    cublasLtHandle_t cublasltHandle_;
#ifdef SPARSITY_ENABLED
    cusparseLtHandle_t cusparseLtHandle_;
    bool is_spmm_compressed = false;
#endif
    std::mutex* cublas_wrapper_mutex_;
    ft::cublasAlgoMap* cublas_algo_map_;
    struct cudaDeviceProp prop_;
    ft::ParallelGptWeight<T> gpt_weights_;
};

class GptOp: public th::jit::CustomClassHolder {
public:
    GptOp(const int64_t head_num,
          const int64_t size_per_head,
          const int64_t inter_size,
          const int64_t layer_num,
          const int64_t vocab_size,
          const int64_t start_id,
          const int64_t end_id,
          const bool sparse,
          const vector<th::Tensor> weights);

    ~GptOp();

    vector<th::Tensor> forward(th::Tensor input_ids,
                               th::Tensor input_lengths,
                               const int64_t output_len,
                               const int64_t beam_width,
                               const int64_t top_k,
                               const double top_p,
                               const double beam_search_diversity_rate,
                               const double temperature,
                               const double len_penalty,
                               const double repetition_penalty,
                               const int64_t random_seed,
                               const int64_t return_cum_log_probs = 0);

private:
    const at::ScalarType st_;
    IFGpt* ftgpt;
    std::vector<th::Tensor> weights;
};

}  // namespace torch_ext
