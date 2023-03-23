# CodeGeeX FasterTransformer

This repository provides the fastertrasformer implementation of [CodeGeeX](https://github.com/THUDM/CodeGeeX) model.

## Get Started
First, download and setup the following docker environment, replace ```<WORK_DIR>``` by the directory of this repo:
```
docker pull nvcr.io/nvidia/pytorch:21.11-py3
docker run -p 9114:5000 --cpus 12 --gpus '"device=0"' -it -v <WORK_DIR>:/workspace/codegeex-fastertransformer --ipc=host  --name=test nvcr.io/nvidia/pytorch:21.11-py3
```
Second, install following packages in the docker:
```
pip3 install transformers
pip3 install sentencepiece
cd codegeex-fastertransformer
sh make_all.sh  # Remember to specify the DSM version according to the GPU.
```
Then, convert the initial checkpoint (download [here](https://models.aminer.cn/codegeex/download/request)) to FT version using ```get_ckpt_ft.py```. 

Finally, run ```api.py``` to start the server and run ```post.py``` to send request:
```
nohup python3 api.py > test.log 2>&1 &
python3 post.py
```

Before run int8 batch inference, need to run 
```
sh convert.sh
```
to convert the model from fp16 into int8. The path in convert.sh need to be set to the right model path.


## Inference performance

The following figure compares the performances of pure Pytorch, Megatron and FasterTransformer under INT8 and FP16.
The fastest implementation is INT8 + FastTrans, and the average time of generating a token <15ms.

<div align=center><img width=100% src="docs/images/inference_performance.png"/></div>

## Liscense

Our code is licensed under the [Apache-2.0 license](LICENSE).
