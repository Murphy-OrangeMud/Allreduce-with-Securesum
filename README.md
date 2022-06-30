# A proposed Machine Learning System for Federated Learning based on Secure Multi-party Computing protocol

基于安全多方计算的联邦学习系统

## 原始数据和可视化代码
毕业设计只涉及到了单机版本在容器上进行实验，分别对python版本和C++版本（更改了Gloo后端）进行测试。
Python版本的数据位于`visualization.py`中，C++版本的数据位于`visualization_cpp.py`中。

## Requirements
pytorch source code，分支release/1.11
torchvision source code，任意分支

若测试gloo版本，运行
```
cp gloo/allreduce.h pytorch/third_party/gloo/gloo
cp gloo/allreduce.cc pytorch/third_party/gloo/gloo
cp gloo/ProcessGroupGloo.cpp pytorch/torch/csrc/distributed/c10d
```

### 单机
单机测试的方法是使用Docker。
```
docker network create $NETWORK_NAME
```
#### python版本
```
docker build . -t $IMAGE_NAME -f Dockerfiles/Dockerfile.python
```
#### gloo版本
```
docker build . -t $IMAGE_NAME -f Dockerfile/Dockerfile.gloo
```
### 多机
需要安装conda。
```
pip install opacus
python download_dataset.py
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake typing_extensions future six requests dataclasses
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
```
在pytorch和vision目录下分别运行
```
python setup.py install
```

## 测试
### 单机
在`$WORLD_SIZE`个不同的终端上分别运行：
```
docker run -t --name hello_$RANK --rm --network $NETWORK_NAME $IMAGE_NAME python benchmark.py --rank $RANK --init_method tcp://hello_0:29500 --world_size $WORLD_SIZE --dataset $DATASET [--baseline] [--differential_privacy]
```

注：
1. `$RANK`从0到`$WORLD_SIZE-1`
2. $DATASET当前支持：mnist、cifar10、cifar100
3. 在gloo后端不更改的前提下，加上--baseline参数指的是测试裸的ring all-reduce
4. 在gloo后端不更改的前提下，加上--differential_privacy指的是测试查分隐私算法
5. 如果是测试gloo版本，必须加上--baseline参数
6. hello_$RANK（hello_0, hello_1, ...)为容器名称，可以改成其他名字，init_method里的hello_0进行相应更改即可

### 多机
尚未开始运行，但在设计中，在每个节点上运行：
```
python benchmark.py --rank $RANK --init_method tcp://$MASTER_IP:29500 --world_size $WORLD_SIZE --dataset $DATASET [--baseline] [--differential_privacy]
```
注释同单机版本。$MASTER_IP指的是主节点的IP地址。
