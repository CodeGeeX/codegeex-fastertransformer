import json
import time
import random
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
import examples.pytorch.codegeex.utils.tokenizer as codetokenizer
from examples.pytorch.codegeex.utils.codegeex import CODEGEEX

tokenizer = codetokenizer.CodeGeeXTokenizer(
    tokenizer_path="./examples/pytorch/codegeex/utils/tokenizer", 
    mode="codegeex-13b")

def tokenize(raw_text):
    seq = tokenizer.encode_code(raw_text)
    return  seq


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
parser.add_argument('--top_k', type=int, default=0,
                    help='top k candidate num')
parser.add_argument('--top_p', type=float, default=1.0,
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
parser.add_argument('--end_id', type=int, default=50256,
                    help='end token id.')
parser.add_argument('--max_batch_size', type=int, default=1,
                    help='max batch size.')
parser.add_argument('--repetition_penalty', type=float, default=1., # default to 1
                    help='repetition penalty')
parser.add_argument('--min_length', type=int, default=0,
                        help='A minimum number of tokens to generate')
parser.add_argument('--max_seq_len', type=int, default=2048,
                    help='max sequence length for position embedding table.')
parser.add_argument('--data_type', type=str, choices=['fp16', 'int8'], default='int8')
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
head_num = args.head_num
output_len = args.output_len
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
min_length = args.min_length
return_cum_log_probs = args.return_cum_log_probs
return_output_length = return_cum_log_probs > 0

# Prepare model.
end_id = tokenizer.eos_token_id
codegeex = CODEGEEX(head_num, size_per_head, vocab_size, start_id, end_id, layer_num,
          max_seq_len, tensor_para_size, pipeline_para_size, lib_path=args.lib_path, dtype=args.data_type)

if not codegeex.load(ckpt_path=args.ckpt_path):
    print("[WARNING] Checkpoint file not found. Model loading is skipped.")
    print("after load")
    exit()
    
if args.is_fix_random_seed == True:
    random_seed_tensor = torch.zeros([1], dtype=torch.int64)
else:
    random_seed_tensor = torch.randint(0, 10000, size=[1], dtype=torch.int64)
    
    
def pad_batch(batch, pad_id, seq_length):
    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        if context_length < seq_length:
            tokens.extend([pad_id] * (seq_length - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def process_code(contexts,output_len,top_k,top_p,beam_search_diversity_rate,temperature,len_penalty,repetition_penalty,min_length,end_tokens):
    batch_size = max_batch_size
    contexts = contexts * batch_size
    start_ids = [tokenize(q) for q in contexts]
    max_len = max([len(q) for q in start_ids])
    start_ids = [[end_id] * (max_len - len(q)) + q for q in start_ids]
    start_lengths = [len(start_id) for start_id in start_ids] * batch_size
    start_ids = [torch.IntTensor(start_id) for start_id in start_ids] * batch_size
    start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id).cuda()
    start_lengths = torch.IntTensor(start_lengths)
    if top_k==0:
        top_k=None
    else:
        top_k=top_k * torch.ones(size=[len(contexts)], dtype=torch.int32)
    if top_p==0.0:
        top_p=None
    else:
        top_p=top_p * torch.ones(size=[len(contexts)], dtype=torch.float32)
    if args.is_fix_random_seed == True:
        random_seed_tensor = torch.zeros([len(contexts)], dtype=torch.int64)
    else:
        random_seed_tensor = torch.randint(0, 10000, size=[len(contexts)], dtype=torch.int64)
    beam_search_diversity_rate=beam_search_diversity_rate * torch.ones(size=[len(contexts)], dtype=torch.float32)
    temperature=temperature * torch.ones(size=[len(contexts)], dtype=torch.float32)
    len_penalty=len_penalty * torch.ones(size=[len(contexts)], dtype=torch.float32)
    repetition_penalty=repetition_penalty * torch.ones(size=[len(contexts)], dtype=torch.float32)
    min_length=min_length * torch.ones(size=[len(contexts)], dtype=torch.int32)
    with torch.no_grad():
        time1 = time.time()
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
                                min_length,
                                random_seed_tensor,
                                return_output_length,
                                return_cum_log_probs)
        time2 = time.time()
        print("time used", time2 - time1)
        outputs = []
        for i, (context, tokens) in enumerate(zip(contexts, tokens_batch)):
            tokens = tokens.cpu().detach().tolist()[0]
            end_idx = len(tokens)
            print("token len: ", end_idx)
            if tokens[start_lengths[i]] == tokenizer.eos_token_id:
                outputs.append({"context": context, "generated": ""})
            else:
                for k, v in enumerate(tokens[start_lengths[i] + 1:]):
                    if v == tokenizer.eos_token_id :
                        end_idx = k + start_lengths[i] + 1
                        break
            print("seq len: ", end_idx - start_lengths[i])
            update_context = tokenizer.decode_code(tokens[start_lengths[i]:end_idx])
            for end_token in end_tokens:
                epos = update_context.find(end_token)
                if epos != -1:
                    update_context = update_context[:epos]
            outputs.append({
                "context": context,
                "generated": update_context,
            })
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
    torch.cuda.empty_cache()
    res = json.loads(request.data)
    context = res['context']
    max_length = res['max_seq_len']
    if 'top_k' in res:
        top_k = res['top_k']
    else:
        top_k=0
    top_p = res['top_p']

    temperature = res['temperature']
    if 'end_tokens' in res:
        end_tokens = res['end_tokens']
    if 'repetition_penalty' in res:
        repetition_penalty = res['repetition_penalty']
    if 'presence_penalty' in res:
        repetition_penalty = res['presence_penalty']
    result = process_code(context,max_length,top_k,top_p,beam_search_diversity_rate,temperature,len_penalty,repetition_penalty,min_length,end_tokens)
    for r in result:
        print(r)
    return json.dumps(result, ensure_ascii=False)

inputs = [
    "# language: Python\n# write a quick sort function\ndef",
    "# language: Python\n# write a quick sort function\n",
    "# language: Python\n# write a bubble sort function\ndef",
    "# language: Python\n# write a merge sort function\ndef",
]

outputs = process_code(
    contexts=inputs, 
    output_len=output_len,
    top_k=top_k,
    top_p=top_p,
    beam_search_diversity_rate=beam_search_diversity_rate,
    temperature=temperature,
    len_penalty=len_penalty,
    repetition_penalty=repetition_penalty,
    min_length=min_length,
    end_tokens=['<|endoftext|>'],
)
for i, output in enumerate(outputs):
    print(f"========= Generation {i} =========")
    print(f"=== Context:\n{output['context']}")
    print(f"=== Generated:\n{output['generated']}")
    print(f"=== Combined:\n{output['context'] + output['generated']}")

print("after 1 process")
app.run('0.0.0.0',port=5000)
