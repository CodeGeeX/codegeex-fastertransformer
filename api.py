import json
import examples.pytorch.codegeex.utils.tokenizer as codetokenizer
import os 
import time
#os.environ['CUDA_LAUNCH_BLOCKING']='1'
'''
tokenizer_args = dict(
    #tokenizer_type='code_ChineseSPTokenizer',
    tokenizer_type='code_GPT2BPETokenizer',
    task_mask=True,
    block_mask_prob=0.0,
    tokenizer_model_type='code-10b',
)
'''
tokenizer = codetokenizer.CodeGeeXTokenizer(
        tokenizer_path="./examples/pytorch/codegeex/utils/tokenizer", 
        mode="codegeex-13b")
def tokenize(raw_text = "他们要打多久"):
    seq = tokenizer.encode_code(raw_text)
    #if not raw_text.endswith('MASK]'):
    #    seq = seq + [tokenizer.get_command('eos').Id]
    #if not mask_ids_pos:
    #    return tokenize(raw_text+"[gMASK]")
    return  seq

import os
import sys

from examples.pytorch.codegeex.utils.codegeex import CODEGEEX, CODEGEEXWeights

from torch.nn.utils.rnn import pad_sequence
import random
import argparse
import timeit
import torch
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--layer_num', type=int, default=40,
                    help='number of layers')
parser.add_argument('--output_len', type=int, default=2048,
                    help='output sequence length to generate.')
parser.add_argument('--head_num', type=int, default=40,
                    help='head number')
parser.add_argument('--size_per_head', type=int, default=128,
                    help='size per head')
parser.add_argument('--vocab_size', type=int, default=52224,
                    help='vocab size')
parser.add_argument('--beam_width', type=int, default=1,
                    help='beam width for beam search. Using sampling when beam width is 1.')
parser.add_argument('--top_k', type=int, default=1,
                    help='top k candidate num')
parser.add_argument('--top_p', type=float, default=0.0,
                    help='top p probability threshold')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature')
parser.add_argument('--len_penalty', type=float, default=1.,
                    help='len_penalty')
parser.add_argument('--beam_search_diversity_rate', type=float, default=0.,
                    help='beam_search_diversity_rate')
parser.add_argument('--tensor_para_size', type=int, default=1,
                    help='tensor parallel size')
parser.add_argument('--pipeline_para_size', type=int, default=1,
                    help='pipeline parallel size')
parser.add_argument('--ckpt_path', type=str, default='/checkpoints/codegeex/codegeex_13b_ft.pt',
                    help='path to the checkpoint file.')
parser.add_argument('--lib_path', type=str, default='./build/lib/libth_codegeex.so',
                    help='path to the pyt_fastertransformer dynamic lib file.')
parser.add_argument('--start_id', type=int, default=50006,
                    help='start token id.')
parser.add_argument('--end_id', type=int, default=50000,
                    help='end token id.')
parser.add_argument('--max_batch_size', type=int, default=1,
                    help='max batch size.')
parser.add_argument('--repetition_penalty', type=float, default=1., # default to 1
                    help='repetition penalty')
parser.add_argument('--max_seq_len', type=int, default=2048,
                    help='max sequence length for position embedding table.')
parser.add_argument('--data_type', type=str, choices=['fp16', 'int8'], default='fp16')
parser.add_argument('--time', action='store_true',
                    help='whether or not to measure time elapsed.')
parser.add_argument('--sample_input_file', type=str, default=None,
                    help='path to sample input file. If not set, it runs with no context inputs.')
parser.add_argument('--sample_output_file', type=str, default=None,
                    help='path to sample output file.')
parser.add_argument('--is_fix_random_seed', type=bool, default=False,
                    help='is fixing the random seed.')
parser.add_argument('--sparse', action='store_true', dest='sparse',
                    help='Enable sparse matrix multiplication. (Need SM 8.0 or 8.6 and SPARSITY_SUPPORT=ON)')
parser.add_argument('--return_cum_log_probs', type=int, default=0, choices=[0, 1, 2],
                    help='Whether to compute the cumulative log probsbility of sentences.'
                         ' 0: do not return the cumulative log probs '
                         ' 1: return the cumulative log probs of generated sequences'
                         ' 2: return the cumulative log probs of sequences')

args = parser.parse_args()

layer_num = args.layer_num
# output_len = args.output_len
head_num = args.head_num
size_per_head = args.size_per_head
vocab_size = args.vocab_size
beam_width = args.beam_width
top_k = args.top_k
top_p = args.top_p
temperature = args.temperature
len_penalty = args.len_penalty
beam_search_diversity_rate = args.beam_search_diversity_rate
tensor_para_size = args.tensor_para_size
pipeline_para_size = args.pipeline_para_size
start_id = args.start_id
end_id = args.end_id
max_batch_size = args.max_batch_size
max_seq_len = args.max_seq_len
repetition_penalty = args.repetition_penalty
return_cum_log_probs = args.return_cum_log_probs
return_output_length = return_cum_log_probs > 0
print("before codegeex")
# Prepare model.
end_id = tokenizer.eos_token_id
print("end_id",end_id)
codegeex = CODEGEEX(head_num, size_per_head, vocab_size, start_id, end_id, layer_num,
          max_seq_len, tensor_para_size, pipeline_para_size, lib_path=args.lib_path, dtype=args.data_type)
print("after codegeex")
if not codegeex.load(ckpt_path=args.ckpt_path):
    print("[WARNING] Checkpoint file not found. Model loading is skipped.")
    print("after load")
    exit()
# if args.data_type == 'fp16':
#     codegeex.half()
#     print("after half")
# elif args.data_type == 'bf16':
#     codegeex.bfloat16()

# if args.sparse:
#     print("in sparse")
#     codegeex.sparse()

if args.is_fix_random_seed == True:
    random_seed = 0
else:
    random_seed = random.randint(0, 100000)
def pad_batch(batch, pad_id, seq_length):
    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        if context_length < seq_length:
            tokens.extend([pad_id] * (seq_length - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths
def process_code(contexts,output_len,top_k,top_p,temperature,repetition_penalty,end_tokens):

    batch_size = max_batch_size
    start_ids = [tokenize(q) for q in contexts]
    '''
    start_lengths = [len(ids) for ids in start_ids]
    input_len = max(start_lengths)

    start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
    start_lengths = torch.IntTensor(start_lengths)
    '''
    tokenized_result = [start_id for start_id in start_ids]
    start_lengths = [len(start_id) for start_id in start_ids]
    start_ids = [torch.IntTensor(start_id  ) for start_id in start_ids] * batch_size
    #start_ids = [start_id  for start_id in start_ids] * batch_size
    print("start_ids",start_ids)
    input_len = max(start_lengths)
    #start_ids,_ = pad_batch(start_ids, tokenizer.eos_token_id, 2048)
    #start_ids = torch.IntTensor(start_ids)
    #start_ids = torch.cuda.LongTensor(start_ids)
    start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id).cuda()
    start_lengths = torch.IntTensor(start_lengths)
    if args.is_fix_random_seed == True:
        random_seed = 0
    else:
        random_seed = random.randint(0, 100000)

    with torch.no_grad():
        tmp_list = list(range(start_lengths))
        print("id",start_ids)
        print(start_lengths)
        time1=time.time()
        tokens_batch = codegeex(start_ids,
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
                           return_output_length,
                           return_cum_log_probs)
        time2=time.time()
        print("time used",time2-time1)
        outputs = []
        #start_lengths = 0
        for i, (context, tokens) in enumerate(zip(contexts, tokens_batch)):
            tokens = tokens.cpu().detach().tolist()[0]
            # print("tokens: ",tokens)
            end_idx = len(tokens)
            print("token len: ",end_idx)
            if tokens[start_lengths]==tokenizer.eos_token_id:
                outputs.append({"context":context,"generated":""})
            else:
                for k,v in enumerate(tokens[start_lengths+1:]):
                    if v == tokenizer.eos_token_id :
                        end_idx = k+start_lengths+1
                        break
            # print("step_generated",tokenizer.DecodeIds(tokenized_result[i][:mask_pos_[0]]))
            # print(tokenizer.DecodeIds(tokens[start_lengths:end_idx]))
            # print(tokenizer.DecodeIds(tokenized_result[i][mask_pos_[0]+1:]))
            print("seq len: ",end_idx-start_lengths)
            update_context = tokenizer.decode_code(
                                           tokens[start_lengths:end_idx] )
            for end_token in end_tokens:
                epos = update_context.find(end_token)
                if epos != -1:
                    update_context = update_context[:epos]
            outputs.append({"context":context,"generated":update_context})
        return outputs

def generate_res(result):
        context = result["context"]
        res = result["generated"]
        inputTokenNum = len(tokenizer.EncodeAsIds(context).tokenization)
        outputTokenNum = len(tokenizer.EncodeAsIds(res).tokenization)
        totalTokenNum = inputTokenNum + outputTokenNum
        data = {}
        return_dict = {}
        return_dict['code'] = 200
        return_dict['msg'] = '成功'
        data['inputText'] = context
        data['outputText'] = res
        data['inputTokenNum'] = inputTokenNum
        data['outputTokenNum'] = outputTokenNum
        data['totalTokenNum'] = totalTokenNum
        return_dict['data'] = data
        return return_dict


from flask import Flask, request

app = Flask(__name__)

@app.route("/code",methods=['POST'])
def hardPromptWrapper():
    res = json.loads(request.data)
    context = res['context']
    max_length = res['max_seq_len']
    if 'top_k' in res:
        top_k = res['top_k']
    else:
        top_k=3
    top_p = res['top_p']
    temperature = res['temperature']
    if 'end_tokens' in res:
        end_tokens = res['end_tokens']
    if 'repetition_penalty' in res:
        repetition_penalty = res['repetition_penalty']
    if 'presence_penalty' in res:
        repetition_penalty = res['presence_penalty']
    result = process_code([context],max_length,top_k,top_p,temperature,repetition_penalty,end_tokens)
    print(result[0])
    #return result[0]
    #return_dict = generate_res(result[0])
    #print(return_dict)
    #print(json.dumps(return_dict))
    return json.dumps(result[0], ensure_ascii=False)


#print(process_code(["不要让我们决定要打多久，他们要打多久[gMASK]"],900))
print(process_code(["#language python\n # sort method"],900,3,0.9,0.9,2.0,['</s>']))

print("after 1 process")
app.run('0.0.0.0',port=5000)
