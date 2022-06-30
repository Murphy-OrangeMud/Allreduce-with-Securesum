import os
from random import Random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp 
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
import argparse
from opacus import PrivacyEngine
from python.allreduce import allreduce_chunk

import time

current_time = 0

BASELINE_PRIVACY = False
BASELINE = True
BENCH_ACCURACY = True
DATASET = "mnist"

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def convnet(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, num_classes, bias=True),
    )


class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(train=True):
    if DATASET == "mnist":
        dataset = datasets.MNIST('./data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    elif DATASET == "cifar10":
        dataset = datasets.CIFAR10('./data', train=True, download=True,
                                transform=transform_train)
    elif DATASET == "cifar100":
        dataset = datasets.CIFAR100('./data', train=True, download=True,
                                transform=transform_train)
    
    size = dist.get_world_size()
    bsz = 128 / float(size)
    bsz = int(bsz)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = DataLoader(partition, batch_size=bsz, shuffle=True)
    return train_set, bsz


def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()

    if DATASET == "mnist":
        model = Net()
    elif DATASET == "cifar10":
        model = convnet(10)
    elif DATASET == "cifar100":
        model = convnet(100)

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))

    if BASELINE_PRIVACY:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_set = privacy_engine.make_private(
            module=model, 
            optimizer=optimizer,
            data_loader=train_set,
            noise_multiplier=1.1, 
            max_grad_norm=1.0,)

    if DATASET == "mnist":
        criterion = nn.NLLLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    count_time(init=True)
    for epoch in range(10):
        epoch_loss = 0.0
        cnt = 0
        correct = 0
        for data, target in train_set:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)

            # record accuracy
            _, pred = output.topk(5)
            pred = pred.t()
            correct += pred.eq(target.view(1, -1)).expand_as(pred).sum().item()

            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
            cnt += 1
        count_time("One epoch")
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': Loss: ', epoch_loss / num_batches, ' Top-5 Accuracy: ', 100. * correct / len(train_set.dataset))

    # test
    if BENCH_ACCURACY:
        if rank == 0:
            if DATASET == "mnist":
                test_set = datasets.MNIST("./data", train=False, download=True,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
            elif DATASET == "cifar10":
                test_set = datasets.CIFAR10('./data', train=False, download=True,
                                            transform=transform_test)
            elif DATASET == "cifar100":
                test_set = datasets.CIFAR100('./data', train=True, download=True,
                                            transform=transform_test)
                
            test_set = DataLoader(test_set, batch_size=128)
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_set:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    _, pred = output.topk(5)
                    pred = pred.t()
                    correct += pred.eq(target.view(1, -1)).expand_as(pred).sum().item()

            test_loss /= len(test_set.dataset)

            print('Test set: Average loss: {:.4f}, Top-5 Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss, correct, len(test_set.dataset), 100. * correct / len(test_set.dataset)
                ))


def count_time(msg="", init=False):
    global current_time
    if init:
        current_time = time.time_ns()
    else:
        print(msg, " time: ", (time.time_ns() - current_time) / 1e9)
        current_time = time.time_ns()


def init_process(rank, size, init_method, fn, backend='nccl' if torch.cuda.is_available() else 'gloo'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=rank, world_size=size, init_method=init_method)
    fn(rank, size)


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if BASELINE:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        else:
            allreduce_chunk(param.grad.data, param.grad.data)
        param.grad.data /= size



parser = argparse.ArgumentParser(description="Distributed training")
parser.add_argument('--rank', type=int, required=True)
parser.add_argument('--init_method', default='tcp://127.0.0.1:29500')
parser.add_argument('--world_size', type=int, default=3)
parser.add_argument('--dataset', default='mnist', choices=['cifar10', 'cifar100', 'mnist'])
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--differential_privacy', action='store_true')
args = parser.parse_args()


if __name__ == "__main__":
    rank = args.rank
    init_method = args.init_method
    size = args.world_size
    DATASET = args.dataset
    BASELINE = args.baseline
    BASELINE_PRIVACY = args.differential_privacy
    # BENCH_ACCURACY = args.accuracy
    processes = []
    mp.set_start_method("spawn")

    p = mp.Process(target=init_process, args=(rank, size, init_method, run))
    p.start()
    
    p.join()

