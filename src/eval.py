import os
import argparse
import logging
import warnings
import random

import numpy as np
import torch
from torch.utils.data import dataloader
from tqdm import tqdm

import open_clip
import utils
import datasets
import model
import test

from torch.cuda.amp import autocast as autocast, GradScaler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings("ignore")
torch.set_num_threads(2)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument('--dataset', default='dress', help="data set type")
parser.add_argument('--fashioniq_split', default='val-split')
parser.add_argument('--fashioniq_path', default=os.path.join(DATA_ROOT, 'FashionIQ') + os.sep)
parser.add_argument('--shoes_path', default=os.path.join(DATA_ROOT, 'Shoes'))
parser.add_argument('--fashion200k_path', default=os.path.join(DATA_ROOT, 'Fashion200k'))

parser.add_argument('--optimizer', default='adamw')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument('--hidden_dim', type=int, default=512)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--clip_lr', type=float, default=1e-6)

parser.add_argument('--backbone', type=str, default='ViT-B-32')

parser.add_argument('--lr_decay', type=int, default=8)
parser.add_argument('--lr_div', type=float, default=0.1)
parser.add_argument('--max_decay_epoch', type=int, default=10)
parser.add_argument('--tolerance_epoch', type=int, default=6)

parser.add_argument('--model_dir', default=os.path.join(PROJECT_ROOT, 'outputs', 'eval'), help="Directory containing params.json")
parser.add_argument('--ckpt', default='', help="ckpt path")

parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=6)

args = parser.parse_args()

if args.backbone == 'ViT-H-14':
    args.pt_path = os.path.join(DATA_ROOT, 'pretrain', 'CLIP-ViT-H-14-laion2B-s32B-b79K', 'open_clip_pytorch_model.bin')
elif args.backbone == 'ViT-B-32':
    args.pt_path = os.path.join(DATA_ROOT, 'pretrain', 'CLIP-ViT-B-32-laion2B-s34B-b79K', 'open_clip_pytorch_model.bin')
else:
    args.pt_path = os.path.join(DATA_ROOT, 'pretrain', 'CLIP-ViT-B-32-laion2B-s34B-b79K', 'open_clip_pytorch_model.bin')


def load_dataset():
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.backbone, pretrained=args.pt_path)
    if args.dataset in ['dress', 'shirt', 'toptee']:
        print('Loading FashionIQ-{} dataset'.format(args.dataset))
        fashioniq_dataset = datasets.FashionIQ(path = args.fashioniq_path, category = args.dataset, transform = [preprocess_train, preprocess_val], split = args.fashioniq_split)
        return [fashioniq_dataset]
    elif args.dataset == 'shoes':
        print('Reading shoes')
        shoes_dataset = datasets.Shoes(path = args.shoes_path, transform = [preprocess_train, preprocess_val])
        return [shoes_dataset]
    elif args.dataset == 'fashion200k':
        print('Reading fashion200k')
        fashion200k_dataset = datasets.Fashion200k(path = args.fashion200k_path, split = 'train', transform = [preprocess_train, preprocess_val])
        fashion200k_testset = datasets.Fashion200k(path = args.fashion200k_path, split = 'test', transform = [preprocess_train, preprocess_val])
        return [fashion200k_dataset, fashion200k_testset]


def set_bn_eval(m): 
    classname = m.__class__.__name__ 
    if classname.find('BatchNorm2d') != -1: 
        m.eval() 

def create_model_and_optimizer():
    ITFD_model = model.ITFD(args, args.hidden_dim, args.dropout_rate)

    if args.ckpt:   
        print(args.ckpt)
        ckpt = torch.load(args.ckpt)
        ITFD_model.load_state_dict(ckpt, strict=True)

    ITFD_model.cuda()

    return ITFD_model

def eval(model, dataset_list):
    model.eval()
    with torch.no_grad():
        if args.dataset == 'fashion200k':
            # load_dataset() 返回 [trainset, testset]
            fashion200k_testset = dataset_list[-1]
            t = test.test_fashion200k_dataset(args, model, fashion200k_testset)
        else:
            # FashionIQ(dress/shirt/toptee) 或 shoes
            t = test.test(args, model, dataset_list[0], args.dataset)
        print(t)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    dataset_list = load_dataset()
    model = create_model_and_optimizer()
    eval(model, dataset_list)
