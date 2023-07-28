import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision import transforms, utils, models
import torch.nn as nn
import argparse
import time
import random
from data import get_data
from model import openset_convnet
from train import pretrain, train, validate
import datetime


def main_workers(args):
    exp_name = f'{datetime.date.today().strftime("%m%d%Y")}_lr{args.lr}_abcd_{args.alpha}_{args.beta}_{args.gamma}_{args.delta}'
    print(exp_name)
    
    train_loader, test_loader = get_data(args)
    model = openset_convnet().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if validate and (args.checkpoint !=None):
        try :
            state_dict = torch.load(args.checkpoint)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            assert missing_keys == [] and unexpected_keys == []
        except :
            ValueError(f"Cannot load weight from {args.checkpoint}")       
    else :
        model = pretrain(args, train_loader, optimizer, model)
        model = train(args, train_loader, optimizer, model)
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
  
  parser.add_argument('--data_path', default='./data')
  parser.add_argument('--checkpoint', default=None)
  parser.add_argument('--validate', type=bool, default=False)
  parser.add_argument('--save_model', type=bool, default=True)
  parser.add_argument('--save_path', default='./results')

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
