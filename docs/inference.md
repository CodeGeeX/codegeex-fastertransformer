说明：机器没有的时候需要拉取docker，新建立的docker退出，然后再进，这样保证docker不会自己关掉。然后make，make之后需要装一些包，然后启动服务

sudo docker pull nvcr.io/nvidia/pytorch:21.11-py3

sudo docker run -p 9114:5000 --cpus 12 --gpus '"device=7"' -it -v /data/glm-fastertransformer:/workspace/FasterTransformer --ipc=host  --name=fastseo7p nvcr.io/nvidia/pytorch:21.11-py3

(-p 是映射的port，-v是映射文件夹路径，--gpus指定gpu，--cpus指定最多cpu个数，--name指定名字，以上根据需要修改）

exit

sudo docker exec -it *dockerid* bash

sh make_all.sh

pip3 install sentencepiece

nohup python3 api.py > log 2>&1 &
