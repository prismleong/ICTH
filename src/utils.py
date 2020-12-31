import random
import argparse
import os

import numpy as np
import timm

import torch
from torch.optim import Adam, AdamW, RMSprop, SGD
from torch.utils.data import DataLoader
from torchvision.datasets import *
import torchvision.transforms as transforms
import torchdata as td

from adamp import AdamP
from radam import RAdam


ICTH_PATH = "\\".join(os.path.abspath(__file__).split('\\')[:-2])
print(ICTH_PATH)
DATA_PATH = os.path.join(ICTH_PATH, 'data')

if "data" not in os.listdir(ICTH_PATH):
    os.mkdir(DATA_PATH)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset", type=str, default='cifar10', choices=[
                        'cifar10', 'cifar100'], help="dataset name in torchvision classificaion dataset")
    parser.add_argument("-ep", "--epochs", type=int, default=10)
    parser.add_argument("-bs", "--batch_size", type=int, default=32)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0)
    parser.add_argument("-sd", "--scheduler", type=str,
                        default="", choices=["", "step", "cosine"])
    parser.add_argument("-seed", "--seed", type=int, default=2020)
    parser.add_argument("-md", "--model", type=str, default="efficientnet_b0",
                        choices=timm.list_models(), help="model name in timm models list")
    parser.add_argument("-wt", "--warmup_type", type=str,
                        default="", choices=["", "linear", "exponential", 'radam'])
    parser.add_argument("-ws", "--warmup_step", type=float, default=0.0)
    parser.add_argument("-opt", "--optimizer", type=str,
                        default='adam', choices=['adam', 'adamw', 'rmsprop', 'sgd', 'radam', 'adamp'])
    parser.add_argument("-rt", "--repeat_times", type=int, default=1,
                        help="how many times of training and testing will be repeated given a group of hyperparameters")
    args = parser.parse_args()

    dataset_dict = {
        'cifar10': CIFAR10,
        'cifar100': CIFAR100
    }
    opt_dict = {
        'adam': Adam,
        'adamw': AdamW,
        'rmsprop': RMSprop,
        'sgd': SGD,
        'radam': RAdam,
        'adamp': AdamP
    }
    args.dataset = dataset_dict[args.dataset]
    args.optimizer = opt_dict[args.optimizer]
    return args


def get_classes_num(args):
    if args.dataset is CIFAR10:
        return 10
    elif args.dataset is CIFAR100:
        return 100


def get_model(args):
    num_classes = get_classes_num(args)
    model = timm.create_model(args.model, num_classes=num_classes)
    return model


def get_dataloader(args, train=True):
    batch_size = args.batch_size

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = td.datasets.WrapDataset(args.dataset(root=DATA_PATH, train=train, download=True,transform=transform)).cache(td.cachers.Memory())
    # pin_memory = True, when ram is sufficient
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=train, pin_memory=True)
    return dataloader
