import os 
import argparse
import logging
import warnings 
import random

from torch.utils.data import Dataset, Subset

import numpy as np 
import torch 
import torch.optim as optim 
from torch.autograd import Variable
from torch.utils.data import dataloader

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
parser.add_argument('--dataset', default = 'dress', help = "data set type")
parser.add_argument('--fashioniq_split', default = 'val-split')
parser.add_argument('--fashioniq_path', default = os.path.join(DATA_ROOT, 'FashionIQ') + os.sep)
parser.add_argument('--shoes_path', default = os.path.join(DATA_ROOT, 'Shoes'))
parser.add_argument('--fashion200k_path', default = os.path.join(DATA_ROOT, 'Fashion200k'))

parser.add_argument('--optimizer', default = 'adamw')
parser.add_argument('--batch_size', type=int, default=32)
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
parser.add_argument('--tolerance_epoch', type=int, default=5)

 
parser.add_argument('--model_dir', default=os.path.join(PROJECT_ROOT, 'outputs', 'checkpoints'),
                    help="Directory containing params.json")

parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--i', type=str, default='0')

####
parser.add_argument('--aug', type=str, default='origin', help = "origin / IDC")
####

args = parser.parse_args()
if args.backbone == 'ViT-H-14':
    args.pt_path = os.path.join(DATA_ROOT, 'pretrain', 'CLIP-ViT-H-14-laion2B-s32B-b79K', 'open_clip_pytorch_model.bin')
elif args.backbone == 'ViT-B-32':
    args.pt_path = os.path.join(DATA_ROOT, 'pretrain', 'CLIP-ViT-B-32-laion2B-s34B-b79K', 'open_clip_pytorch_model.bin')
else:
    args.pt_path = os.path.join(DATA_ROOT, 'pretrain', 'resnet50_clip', 'open_clip_pytorch_model.bin')

def load_dataset():
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(args.backbone, pretrained=args.pt_path)
    if args.dataset in ['dress', 'shirt', 'toptee']:
        print('Loading FashionIQ-{} dataset'.format(args.dataset))
        fashioniq_dataset = datasets.FashionIQ(path = args.fashioniq_path, category = args.dataset, transform = [preprocess_train, preprocess_val], split = args.fashioniq_split, aug=args.aug)
        # print("len: ", len(fashioniq_dataset))
        return [fashioniq_dataset]
    elif args.dataset == 'shoes':
        print('Reading shoes')
        shoes_dataset = datasets.Shoes(path = args.shoes_path, transform = [preprocess_train, preprocess_val])
        # print("shoes_dataset: ", len(shoes_dataset))
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
    ITFD_model.cuda(0)

    params = list(ITFD_model.named_parameters())
    param_group = [
        {'params': [p for n, p in params if any(nd in n for nd in ['clip'])], 'lr': args.clip_lr},
        {'params': [p for n, p in params if not any(nd in n for nd in ['clip'])], 'lr': args.lr},
    ]
    optimizer = torch.optim.AdamW(param_group, lr=args.lr, weight_decay = args.weight_decay)
    return ITFD_model, optimizer


def train(model, optimizer, dataloader, scaler):
    model.train()
    model.apply(set_bn_eval)
    summ = []
    loss_avg = utils.RunningAverage()
    for i, data in enumerate(dataloader):

        target_img = data['target_img_data'].cuda(0)
        textual_query = data['textual_query']
        visual_query = data['visual_query'].cuda(0)

        # IDC
        # t1 = data[0]['target_img_data'].cuda(0)
        # t2 = data[1]['target_img_data'].cuda(0)
        # m1 = data[0]['textual_query']
        # m2 = data[1]['textual_query']
        # q1 = data[0]['visual_query'].cuda(0)
        # q2 = data[0]['visual_query'].cuda(0)
        # target_img = torch.cat((t1, t2), dim=0)
        # textual_query = m1 + m2
        # visual_query = torch.cat((q1, q2), dim=0)
        # print("target_img: ", target_img.shape)
        # print("target_img2: ", target_img2.shape)
        # print("textual_query: ", len(textual_query))
        # print("textual_query2: ", len(textual_query2))
        # print("visual_query: ", visual_query.shape)
        # print("visual_query2: ", visual_query2.shape)

        optimizer.zero_grad()
        with autocast():
            loss = model.compute_loss(textual_query, visual_query, target_img)
            total_loss = loss['ranking']

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()     

        if i % args.save_summary_steps == 0:
            summary_batch = {}
            summary_batch['total_loss'] = total_loss.item()
            summ.append(summary_batch)
        loss_avg.update(total_loss.item())


def train_and_evaluate(model, optimizer, dataset_list):
    if args.dataset == 'fashion200k':
        fashion200k_testset = dataset_list.pop(-1)
    trainloader = dataloader.DataLoader(dataset_list[0],
                                batch_size = args.batch_size,
                                shuffle = True,
                                num_workers=args.num_workers)
    print("trainloader: ", len(trainloader))


    best_score = float('-inf')
    tolerance = 0
    scaler = GradScaler()
    epoches = args.num_epochs

    for epoch in range(epoches):

        tolerance = tolerance + 1
        if epoch != 0 and (epoch+1) % args.lr_decay == 0 and epoch < args.max_decay_epoch:
            for g in optimizer.param_groups:
                g['lr'] *= args.lr_div

        logging.info("Epoch {}/{}".format(epoch + 1, epoches))
        train(model, optimizer, trainloader, scaler)

        current_score = 0
        if tolerance < args.tolerance_epoch:
            if args.dataset in ['dress', 'shirt', 'toptee', 'shoes']:
                with torch.no_grad():
                    t = test.test(args, model, dataset_list[0], args.dataset)
                logging.info(t)
                current_score = current_score + t[1][1] + t[2][1]

            
            elif args.dataset in ['fashion200k']:
                t = test.test_fashion200k_dataset(args, model, fashion200k_testset)
                logging.info(t)
                current_score = current_score + t[0][1] + t[1][1] + t[2][1]
            
            if current_score > best_score:
                best_score = current_score
                tolerance = 0
                best_json_path = os.path.join(
                    args.model_dir, "{}_{}_metrics_best.json".format(args.dataset, args.i))
                test_metrics = {}
                for metric_name, metric_value in t:
                    test_metrics[metric_name] = metric_value

                utils.save_dict_to_json(test_metrics, best_json_path)
                # save model
                torch.save(model.state_dict(), os.path.join(args.model_dir, "{}_{}_best_model.pt".format(args.dataset, args.i)))
        else:
            break


if __name__ == '__main__':

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if args.dataset in ['dress', 'shirt', 'toptee', 'shoes']:
        utils.set_logger(os.path.join(args.model_dir, '{}_{}_{}_train.log'.format(args.dataset, args.fashioniq_split, args.backbone)))
    else:
        utils.set_logger(os.path.join(args.model_dir, '{}_{}_train.log'.format(args.dataset, args.backbone)))
    logging.info('Loading the datasets and model...')

    dataset_list = load_dataset()
 
    model, optimizer = create_model_and_optimizer()
    logging.info("Starting train for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, optimizer, dataset_list)
