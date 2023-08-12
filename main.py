import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision import transforms, utils, models
from data import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR
import torch.nn as nn
import argparse
import time
import random
from data import get_data
import models
from train import pretrain, train, validate
import datetime
from models.wide_resnet import Wide_ResNet
from split import splits_AUROC, splits_F1


def run_fold(args, fold):
    print(f'FOLD {fold} starts')
    data_path = os.path.join(args.data_root, args.dataset)
    if 'mnist' in args.dataset:
        Data = MNIST_OSR(known=args.cls_known, dataroot=data_path, batch_size=args.batch_size,img_size=args.img_size)
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar10' == args.dataset:
        Data = CIFAR10_OSR(known=args.cls_known, dataroot=data_path, batch_size=args.batch_size,img_size=args.img_size)
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'svhn' in args.dataset:
        Data = SVHN_OSR(known=args.cls_known, dataroot=data_path, batch_size=args.batch_size,img_size=args.img_size)
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    else:
        Data = Tiny_ImageNet_OSR(known=args.cls_known, dataroot=data_path, batch_size=args.batch_size,img_size=args.img_size)
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader

    if args.backbone == 'WideResnet':
        backbone = Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=len(args.cls_known)) # proser default setting
    elif args.backbone == 'Toy':
        backbone = None

    model = models.__dict__[args.model](args,backbone).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=args.lr*0.01)
    if validate and (args.checkpoint !=None):
        try :
            state_dict = torch.load(args.checkpoint)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            assert missing_keys == [] and unexpected_keys == []
        except :
            ValueError(f"Cannot load weight from {args.checkpoint}")       
    else :
        model = pretrain(args, trainloader, optimizer, model)
        model = train(args, trainloader, optimizer, model, scheduler, params=[args.a, args.b, args.c, args.d])
    model.eval()
    in_acc ,ood_acc, auc = validate(args, model, testloader, outloader)
    txt_name = f'{args.model}_{args.dataset}_{args.backbone}.txt'
    with open(txt_name, 'a' if os.path.isfile(txt_name) else 'w') as f:
        f.write(f'Fold {fold}  |  closed acc : {in_acc:.5f}  |  open acc : {ood_acc:.5f}  |  AUC : {auc:.5f}\n')

    return in_acc ,ood_acc, auc
  
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_fold', type=int, default=5)
    parser.add_argument('--pretrain_epochs', default=5, type=int)
    parser.add_argument('--train_epochs', default=20, type=int)
    parser.add_argument('--device', default='cuda:0',type=str)
    parser.add_argument('--print_epoch', default=1,type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--data_root', default='../../data')

    parser.add_argument('--param_schedule', type=int, default=5, help='epoch to schedule parameters')
    parser.add_argument('--param_step', type=float, default=0.02, help='epoch to schedule parameters')
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--model', default='doser', help='proser | efficient_proser | doser')
    parser.add_argument('--backbone', default='Toy', help='WideResnet | Toy')
    parser.add_argument('--dataset', type=str, default='mnist', help="mnist | svhn | cifar10 | cifar100(보류) | tiny_imagenet")
    parser.add_argument('--lambda1', default=0, type=float, help='ratio to keep original representation while doing manifold mix-up')
    parser.add_argument('--a', default= 0.5, type=float, help = 'weight for the mix loss')
    parser.add_argument('--b', default= 1, type=float, help='weight for the class loss')
    parser.add_argument('--c', default= 1, type=float, help = 'weight for the mute loss')
    parser.add_argument('--d', default= 0.5, type=float, help = 'weight for the reconloss')

    # val and save
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--validate', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--save_path', default='./results')
    args = parser.parse_args(args=[])

  
    return args
  
if __name__ == "__main__":
    args = arg_parser()

    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    exp_name = f'{datetime.date.today().strftime("%m%d%Y")}_lr{args.lr}_abcd_{args.a}_{args.b}_{args.c}_{args.d}_paramupdate_{args.param_step}_per_{args.param_schedule}epoch_lambda{args.lambda1}'
    print(exp_name)
    cls_config = splits_AUROC

    in_acc_list = []
    ood_acc_list = []
    auc_list=[]
    txt_name = f'{args.model}_{args.dataset}_{args.backbone}.txt'
    with open(txt_name, 'a' if os.path.isfile(txt_name) else 'w') as f:
        f.write(f'===================== {exp_name} =======================\n')
    for i in range(args.num_fold):
        args.cls_known = cls_config[args.dataset][i]
        in_acc ,ood_acc, auc = run_fold(args, i+1)
        in_acc_list.append(in_acc)
        ood_acc_list.append(ood_acc)
        auc_list.append(auc)
    with open(txt_name, 'a' if os.path.isfile(txt_name) else 'w') as f:
        f.write(f'{args.num_fold} Fold Average : closed acc : {sum(in_acc_list)/len(in_acc_list):5f}  |  open acc : {sum(ood_acc_list)/len(ood_acc_list):.5f}  |  AUC : {sum(auc_list)/len(auc_list):.5f}\n')
        f.write(f'================================================================================\n\n\n')