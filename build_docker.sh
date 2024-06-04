docker run --gpus all -it --ipc=host --net=host --name=muxllm -v /users/zyh:/users/zyh nvcr.io/nvidia/pytorch:23.10-py3

# cd /users/zyh/vllm && pip install -r requirements.txt && python setup.py develop # build vllm