import torch
from utils.data import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR
import os

def load_model(args, model):
    try :
        state_dict = torch.load(args.checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        assert missing_keys == [] and unexpected_keys == []
    except :
        ValueError(f"Cannot load weight from {args.checkpoint}")  
    return model

def mute_max(logits):
    min_values, _ = torch.min(logits, dim=1, keepdim=True)
    max_values, _ = torch.max(logits, dim=1, keepdim=True)
    muted = torch.where(logits == max_values, min_values, logits)
    return muted

def param_schedule(a,b,c,d,step):
    return a+step, b, c-step, d-step


def get_loaders(args):
    data_path = os.path.join(args.data_root, args.dataset)
    if 'mnist' in args.dataset:
        Data = MNIST_OSR(known=args.cls_known, dataroot=data_path, batch_size=args.batch_size,img_size=args.img_size)
    elif 'cifar10' == args.dataset:
        Data = CIFAR10_OSR(known=args.cls_known, dataroot=data_path, batch_size=args.batch_size,img_size=args.img_size)
    elif 'svhn' in args.dataset:
        Data = SVHN_OSR(known=args.cls_known, dataroot=data_path, batch_size=args.batch_size,img_size=args.img_size)
    else:
        Data = Tiny_ImageNet_OSR(known=args.cls_known, dataroot=data_path, batch_size=args.batch_size,img_size=args.img_size)
    
    return Data.train_loader, Data.test_loader, Data.out_loader