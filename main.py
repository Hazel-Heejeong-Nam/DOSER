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

def main_workers(args):
    exp_name = f'{datetime.date.today().strftime("%m%d%Y")}_lr{args.lr}_abcd_{args.alpha}_{args.beta}_{args.gamma}_{args.delta}'
    print(exp_name)
    
    cls_config = splits_AUROC
    # 일단 cv없이0번째만
    args.cls_known = cls_config[args.dataset][0]

    print("{} Preparation".format(args.dataset))
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
    # elif 'cifar100' in args.dataset:
    #     Data = CIFAR10_OSR(known=cls_known, dataroot=data_path, batch_size=args.batch_size,img_size=args.img_size)
    #     trainloader, testloader = Data.train_loader, Data.test_loader
    #     out_Data = CIFAR100_OSR(known=options['unknown'], dataroot=data_path, batch_size=args.batch_size, img_size=args.img_size)
    #     outloader = out_Data.test_loader
    else:
        Data = Tiny_ImageNet_OSR(known=args.cls_known, dataroot=data_path, batch_size=args.batch_size,img_size=args.img_size)
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader

    backbone = Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=len(args.cls_known)) # proser default setting
    model = models.__dict__[args.model](args,backbone).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if validate and (args.checkpoint !=None):
        try :
            state_dict = torch.load(args.checkpoint)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            assert missing_keys == [] and unexpected_keys == []
        except :
            ValueError(f"Cannot load weight from {args.checkpoint}")       
    else :
        model = pretrain(args, trainloader, optimizer, model)
        model = train(args, trainloader, optimizer, model)
    model.eval()
    validate(test_loader, model)

    if args.save_model :
      os.makedirs(os.path.join(args.save_path, exp_name+'.pth'), exist_ok=True)
      torch.save(model.state_dict, )
  
def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr', default=0.0003, type=float)
  parser.add_argument('--pretrain_epochs', default=5, type=int)
  parser.add_argument('--train_epochs', default=10, type=int)
  parser.add_argument('--device', default='cuda:0',type=str)
  parser.add_argument('--print_epoch', default=1,type=int)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--img_size', type=int, default=32)
  parser.add_argument('--data_root', default='../../data')
  parser.add_argument('--checkpoint', default=None)
  parser.add_argument('--validate', type=bool, default=False)
  parser.add_argument('--save_model', type=bool, default=True)
  parser.add_argument('--save_path', default='./results')
  parser.add_argument('--model', default='doser', help='proser | efficient_proser | doser')
  parser.add_argument('--backbone', default='WideResnet', help='WideResnet | Basic')
  parser.add_argument('--dataset', type=str, default='mnist', help="mnist | svhn | cifar10 | cifar100(보류) | tiny_imagenet")
  #parser.add_argument('--lambda', default=0, type=float, help='ratio to keep original representation while doing manifold mix-up')
  parser.add_argument('--alpha', default= 0.5, type=float, help = 'weight for the loss1')
  parser.add_argument('--beta', default= 1, type=float, help='weight for the loss2_1')
  parser.add_argument('--gamma', default= 1, type=float, help = 'weight for the loss 2_2')
  parser.add_argument('--delta', default= 0.5, type=float, help = 'weight for the reconloss')
  args = parser.parse_args(args=[])
  return args
  
if __name__ == "__main__":
  args = arg_parser()

  random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed_all(0)

  main_workers(args)
