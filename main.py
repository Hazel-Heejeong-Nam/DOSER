import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision import transforms, utils, models
from utils.data import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR
import torch.nn as nn
import argparse
import time
import random
import models
from train import pretrain, train, validate, get_pic
import datetime
from split import splits_AUROC, splits_F1
from utils import load_model, get_loaders, Wide_ResNet


def run_fold(args, fold):
    print(f'FOLD {fold} starts')
    data_path = os.path.join(args.data_root, args.dataset)
    trainloader, testloader, outloader = get_loaders(args)

    if args.backbone == 'WideResnet':
        backbone = Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=len(args.cls_known)) # proser default setting
    elif args.backbone == 'Toy':
        backbone = None

    model = models.__dict__[args.model](args,backbone).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=args.lr*0.01) if args.lr_schedule == True else None

    if args.validate and (args.checkpoint !=None):
        model = load_model(args,model)  
    else :
        if args.checkpoint !=None and 'pretrain' in args.checkpoint: 
            model = load_model(args,model)  
        else:
            model = pretrain(args, trainloader, optimizer, model)
            if args.save_model :
                os.makedirs(args.save_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.save_path, f'new_pretrain_{args.model}_{args.dataset}_{args.backbone}_epoch{args.pretrain_epochs}_lr{args.lr}_fold{fold}.pth'))
        model = train(args, trainloader, optimizer, model, scheduler, params=[args.a, args.b, args.c, args.d])
        if args.save_model :
            os.makedirs(args.save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_path, f'new_train_{args.model}_{args.dataset}_{args.backbone}_train{args.train_epochs}_lr{args.lr}_schedule_{args.lr_schedule}_abcd_{args.a}_{args.b}_{args.c}_{args.d}_paramupdate_{args.param_step}_per_{args.param_schedule}epoch_fold{fold}.pth'))
    model.eval()
    if args.analysis :
        get_pic(args,model, trainloader)
    in_acc ,ood_acc, avg_acc, auc = validate(args, model, testloader, outloader)
    txt_name = f'{args.model}_{args.dataset}_{args.backbone}.txt'
    with open(txt_name, 'a' if os.path.isfile(txt_name) else 'w') as f:
        f.write(f'Fold {fold}  |  closed acc : {in_acc:.5f}  |  open acc : {ood_acc:.5f}  |  mean acc : {avg_acc:.5f}  |  AUC : {auc:.5f}\n')

    return in_acc ,ood_acc, avg_acc, auc

def main_worker(args):
    exp_name = f'{datetime.date.today().strftime("%m%d%Y")}_lr{args.lr}_abcd_{args.a}_{args.b}_{args.c}_{args.d}_paramupdate_{args.param_step}_per_{args.param_schedule}epoch_lambda{args.lambda1}'
    print(exp_name)
    cls_config = splits_AUROC

    in_acc_list = []
    ood_acc_list = []
    avg_acc_list = []
    auc_list=[]
    txt_name = f'{args.model}_{args.dataset}_{args.backbone}.txt'
    with open(txt_name, 'a' if os.path.isfile(txt_name) else 'w') as f:
        f.write(f'===================== {exp_name} =======================\n')
    for i in range(args.num_fold):
        if args.dataset =='mnist' and args.backbone=='Toy':
            args.cls_known = [1,2,3,4,5,6,7,8,9]
        else : 
            args.cls_known = cls_config[args.dataset][i]
        in_acc ,ood_acc, avg_acc, auc = run_fold(args, i+1)
        in_acc_list.append(in_acc)
        ood_acc_list.append(ood_acc)
        avg_acc_list.append(avg_acc)
        auc_list.append(auc)
    with open(txt_name, 'a' if os.path.isfile(txt_name) else 'w') as f:
        f.write(f'{args.num_fold} Fold Average : closed acc : {sum(in_acc_list)/len(in_acc_list):5f}  |  open acc : {sum(ood_acc_list)/len(ood_acc_list):.5f}  |  mean acc : {sum(avg_acc_list)/len(avg_acc_list):.5f}    |  AUC : {sum(auc_list)/len(auc_list):.5f}\n')
        f.write(f'================================================================================\n\n\n')  

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_fold', type=int, default=1)
    parser.add_argument('--pretrain_epochs', default=30, type=int)
    parser.add_argument('--train_epochs', default=30, type=int)
    parser.add_argument('--device', default='cuda:0',type=str)
    parser.add_argument('--print_epoch', default=1,type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--data_root', default='../../data')


    parser.add_argument('--model', default='doser_nodisentangle', help='Proser | efficient_Proser | doser | doser_nodisentangle ')
    parser.add_argument('--backbone', default='Toy', help='WideResnet | Toy')
    parser.add_argument('--dataset', type=str, default='cifar10', help="mnist | svhn | cifar10 | tiny_imagenet")
    parser.add_argument('--param_schedule', type=int, default=8, help='epoch to schedule parameters')
    parser.add_argument('--param_step', type=float, default=0.03, help='epoch to schedule parameters')
    parser.add_argument('--lr_schedule', type=bool,default=True )


    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--lambda1', default=0.2, type=float, help='ratio to keep original representation while doing manifold mix-up')
    parser.add_argument('--a', default= 0.1, type=float, help = 'weight for the mix loss')
    parser.add_argument('--b', default= 1, type=float, help='weight for the class loss')
    parser.add_argument('--c', default= 1, type=float, help = 'weight for the mute loss')
    parser.add_argument('--d', default= 0.3, type=float, help = 'weight for the reconloss')

    # val and save
    parser.add_argument('--checkpoint', default='results/new_pretrain_doser_cifar10_Toy_epoch30_lr0.0003_fold1.pth', type=str)
    parser.add_argument('--validate', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--save_path', default='./results')
    parser.add_argument('--analysis', type=bool, default=False)
    args = parser.parse_args()

  
    return args
  
if __name__ == "__main__":
    args = arg_parser()

    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    main_worker(args)