from datasets import load_dataset, load_metric
from transformers import AdamW, get_scheduler, get_cosine_schedule_with_warmup
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoImageProcessor, ConvNextForImageClassification

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils import parse_args, tf_make_result_path, seed_fix

from tqdm import tqdm
import os
import json
import wandb
import argparse

dataset_to_imagedict = { 
    "cifar10": 'img',
    "cifar100": 'img',
    "imagenet-1k": 'image', 
}
dataset_to_labeldict = { 
    "cifar10": 'label',
    "cifar100": 'fine_label',
    "imagenet-1k": 'label', 
}

data_path="./"
dataset_name = "imagenet-1k"
dataset_labeldict = dataset_to_labeldict[dataset_name]
dataset_imagedict = dataset_to_imagedict[dataset_name]
dataset = load_dataset(dataset_name , cache_dir=data_path, use_auth_token=True) 
metric = load_metric("accuracy") 