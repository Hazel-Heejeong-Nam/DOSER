import numpy as np
import torch
from PIL import Image
from torchvision.datasets.mnist import read_image_file, read_label_file
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision import transforms, utils, models
import torch.nn as nn
import argparse
import time
import random


class Meter:
    def __init__(self):
        self.list = []
    def update(self, item):
        self.list.append(item)
    def avg(self):
        return torch.tensor(self.list).mean() if len(self.list) else None

def mute_max(logits):
    min_values, _ = torch.min(logits, dim=1, keepdim=True)
    max_values, _ = torch.max(logits, dim=1, keepdim=True)
    muted = torch.where(logits == max_values, min_values, logits)
    return muted
  
def compute_accuracy(logits, labels):
  pred = torch.argmax(logits, dim=1)
  zero_idx= ((labels==0).nonzero(as_tuple=True)[0])
  else_idx= ((labels!=0).nonzero(as_tuple=True)[0])

  if len(zero_idx) == 0:
    zero_acc = None
    else_acc = (pred[else_idx] == labels[else_idx]).type(torch.float).mean().item()*100
  else:
    zero_acc = (pred[zero_idx] == labels[zero_idx]).type(torch.float).mean().item()*100
    else_acc = (pred[else_idx] == labels[else_idx]).type(torch.float).mean().item()*100
  return zero_acc, else_acc


def main_workers(args):
  transform = transforms.Compose([transforms.ToTensor()])
  train_dataset = custom_MNIST(train=True, transform=transform)
  test_dataset = custom_MNIST(train=False, transform=transform)
  
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

  model = openset_convnet().to(args.device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=0)
  class_cri = nn.CrossEntropyLoss()
  reconstruction_loss = nn.MSELoss()

  start_time = time.time()
  for epoch in range(args.pretrain_epochs):
      pre_total_loss = 0
      for idx, (images, labels) in enumerate(train_loader):
  
          optimizer.zero_grad()
          labels = labels.to(args.device) 
          
          id_logits, _, recon = model(images,mode='train')
          class_loss = class_cri(id_logits, labels)
          recon_loss = reconstruction_loss(recon.cpu(), images.view(images.size(0), -1))
          
          pre_loss = class_loss + recon_loss
          pre_loss.backward()
          pre_total_loss += pre_loss.item()
          optimizer.step()
          
      if (epoch+1) % args.print_epoch == 0:
          print(f'Pretrain epoch : {epoch}, average loss : {pre_total_loss/((idx+1)*args.print_epoch):.4f},  time elapsed : {time.time()-start_time:.4f}sec')
          start_time = time.time()
  print('---------------------------------\n Pretraining Done')

  # Training
  start_time = time.time()
  for epoch in range(args.train_epochs):
      total_loss1 = 0
      total_loss2 = 0
      total_loss3 = 0
      total_recon = 0
      for idx, (images, labels) in enumerate(train_loader):
  
          labels = labels.to(args.device) # labels 32  
          ood_label = torch.zeros_like(labels).to(args.device)
          
          optimizer.zero_grad()
          id_logits, ood_logits, recons = model(images, mode='train')
          muted_logits = mute_max(id_logits.clone())
          
          mixing_loss = class_cri(ood_logits, ood_label) 
          class_loss = class_cri(id_logits, labels) 
          muted_loss = class_cri(muted_logits, ood_label)
          recon_loss = reconstruction_loss(recons.cpu(), images.view(images.size(0), -1))
          
          total_loss = mixing_loss * args.alpha + class_loss * args.beta + muted_loss * args.gamma + recon_loss * args.delta
          total_loss.backward()
          total_loss1 += mixing_loss.item()
          total_loss2 += class_loss.item()
          total_loss3 += muted_loss.item()
          total_recon += recon_loss.item()
          optimizer.step()
  
          
      if (epoch+1) % args.print_epoch == 0:
          print(f'Epoch : {epoch}, loss1 : {total_loss1/((idx+1)*args.print_epoch):.4f}, loss2 : {total_loss2/((idx+1)*args.print_epoch):.4f}, \
  loss3 : {total_loss3/((idx+1)*args.print_epoch):.4f}, recon : {total_recon/((idx+1)*args.print_epoch):.4f}, \
  time elapsed : {time.time()-start_time:.4f}sec')
          start_time = time.time()
          
      if (epoch+1)%3 == 0:
          args.alpha += 0.02
          args.gamma -= 0.02
          args.delta -= 0.1
      
  print('---------------------------------\n Training Done')

  # evaluate
  acc_zero = Meter()
  acc_else = Meter()
  
  with torch.no_grad():
    model.eval()
    for i, (data, labels) in enumerate(test_loader):
      # Get 0~9 classification output logits / size : Batch x 10
      logits = model(data)
    
      zero_acc, else_acc = compute_accuracy(logits, labels)
      if zero_acc != None:
        acc_zero.update(zero_acc)
        acc_else.update(else_acc)
      else:
        acc_else.update(else_acc)
  
  print("zero_acc: {:.4f}, else_acc: {:.4f}".format(acc_zero.avg(), acc_else.avg()))
def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr', default=0.0003, type=float)
  parser.add_argument('--pretrain_epochs', default=5, type=int)
  parser.add_argument('--train_epochs', default=10, type=int)
  parser.add_argument('--device', default='cuda:0',type=str)
  parser.add_argument('--print_epoch', default=1,type=int)
  
  #parser.add_argument('--lambda', default=0, type=float, help='ratio to keep original representation while doing manifold mix-up')
  parser.add_argument('--alpha', default= 0.5, type=float, help = 'weight for the loss1')
  parser.add_argument('--beta', default= 1, type=float, help='weight for the loss2_1')
  parser.add_argument('--gamma', default= 1, type=float, help = 'weight for the loss 2_2')
  parser.add_argument('--delta', default= 0.5, type=float, help = 'weight for the reconloss')
  args = parser.parse_args(args=[])
  return args
  
if __name__ == "__main__":
  args = args_parser()

  random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed_all(0)

  main_workers(args)