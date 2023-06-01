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

#include "src/fastertransformer/th_op/codegeex/CodegeexOp.h"

namespace th = torch;
namespace torch_ext {

CodegeexOp::CodegeexOp(const int64_t head_num,
             const int64_t size_per_head,
             const int64_t inter_size,
             const int64_t layer_num,
             const int64_t vocab_size,
             const int64_t start_id,
             const int64_t end_id,
             const bool sparse,
             const int64_t dtype_id,
             const std::vector<th::Tensor> weights,
             const std::vector<th::Tensor> quant_weights,
             const std::vector<th::Tensor> quant_scales):
    st_(weights[0].scalar_type())
{
    for (auto t : weights) {
        CHECK_INPUT(t, st_);
    }

    switch (st_) {
        case at::ScalarType::Float:
            ftcodegeex = new FTCodegeex<float>((size_t)head_num,
                                     (size_t)size_per_head,
                                     (size_t)inter_size,
                                     (size_t)layer_num,
                                     vocab_size,
                                     start_id,
                                     end_id,
                                     sparse,
                                     dtype_id,
                                     weights,
                                     quant_weights,
                                     quant_scales);
            break;
        case at::ScalarType::Half:
            ftcodegeex = new FTCodegeex<half>((size_t)head_num,
                                    (size_t)size_per_head,
                                    (size_t)inter_size,
                                    (size_t)layer_num,
                                    (size_t)vocab_size,
                                    start_id,
                                    end_id,
                                    sparse,
                                    dtype_id,
                                    weights,
                                    quant_weights,
                                    quant_scales);
            break;
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
            ftcodegeex = new FTCodegeex<__nv_bfloat16>((size_t)head_num,
                                             (size_t)size_per_head,
                                             (size_t)inter_size,
                                             (size_t)layer_num,
                                             (size_t)vocab_size,
                                             start_id,
                                             end_id,
                                             sparse,
                                             dtype_id,
                                             weights,
                                             quant_weights,
                                             quant_scales);
            break;
#endif
        default:
            throw std::runtime_error("Wrong Tensor type.");
    }
}

CodegeexOp::~CodegeexOp()
{
    delete ftcodegeex;
}

std::vector<th::Tensor> CodegeexOp::forward(th::Tensor input_ids,
                                       th::Tensor input_lengths,
                                       const int64_t output_len,
                                       const int64_t beam_width,
                                       th::optional<th::Tensor> top_k,
                                       th::optional<th::Tensor> top_p,
                                       th::optional<th::Tensor> beam_search_diversity_rate,
                                       th::optional<th::Tensor> temperature,
                                       th::optional<th::Tensor> len_penalty,
                                       th::optional<th::Tensor> repetition_penalty,
                                       th::optional<th::Tensor> min_length,
                                       th::Tensor random_seed,
                                       const int64_t return_cum_log_probs)
{
    CHECK_TH_CUDA(input_ids);
    CHECK_CONTIGUOUS(input_ids);
    TORCH_CHECK(input_ids.dtype() == torch::kInt32, "input_ids dtype should be int32");
    CHECK_TH_CUDA(input_lengths);
    CHECK_CONTIGUOUS(input_lengths);
    TORCH_CHECK(input_lengths.dtype() == torch::kInt32, "input_lengths dtype should be int32");
    TORCH_CHECK(return_cum_log_probs == 0 || return_cum_log_probs == 1 || return_cum_log_probs == 2,
                "return_cum_log_probs should be"
                " 0 (no return cum_log_probs), "
                " 1 (the cumulative log probs of generated sequences), or"
                " 2 (the cumulative log probs of sequences).")

    const int batch_size = input_ids.size(0);
    const int max_input_length = input_ids.size(1);
    const int total_request_output_len = max_input_length + output_len;
    th::Tensor output_ids = torch::empty({batch_size, beam_width, total_request_output_len},
                                         torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    th::Tensor parent_ids = torch::empty({total_request_output_len, batch_size, beam_width},
                                         torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    th::Tensor sequence_lengths =
        torch::empty({batch_size, beam_width}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
    th::Tensor cum_log_probs =
        torch::empty({batch_size, beam_width}, torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false));

    ftcodegeex->forward(input_ids,
                   input_lengths,
                   output_ids,
                   parent_ids,
                   sequence_lengths,
                   cum_log_probs,
                   (const size_t)output_len,
                   (const size_t)beam_width,
                   top_k,
                   top_p,
                   beam_search_diversity_rate,
                   temperature,
                   len_penalty,
                   repetition_penalty,
                   min_length,
                   random_seed,
                   return_cum_log_probs);
    if (return_cum_log_probs > 0) {
        return std::vector<th::Tensor>{output_ids, sequence_lengths, cum_log_probs};
    }
    return std::vector<th::Tensor>{output_ids, sequence_lengths};
}

}  // namespace torch_ext

static auto fasterTransformerCodegeexTHS =
    torch::jit::class_<torch_ext::CodegeexOp>("FasterTransformer", "CodegeexOp")
        .def(torch::jit::
                 init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool, int64_t, std::vector<th::Tensor>, std::vector<th::Tensor>, std::vector<th::Tensor>>())
        .def("forward", &torch_ext::CodegeexOp::forward);
