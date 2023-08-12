import numpy as np
import torch
from PIL import Image
from torchvision.datasets.mnist import read_image_file, read_label_file
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision import transforms, utils, models
import torch.nn as nn
from .base import Basic


class doser(Basic):
    def __init__(self,args,net):
        super().__init__(args,net)
        self.net = net
        self.content_enc = self.encoder1
        self.style_enc = self.encoder2
        if args.backbone == 'Toy':
            self.decode= nn.Sequential(
                nn.Linear(32+64, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 3072),
                nn.Sigmoid()
            )
        elif args.backbone == 'WideResnet':
            self.decode= nn.Sequential(
                nn.Linear(320+640, 1920),
                nn.ReLU(),
                nn.Linear(1920, 1920),
                nn.ReLU(),
                nn.Linear(1920, 2880),
                nn.ReLU(),
                nn.Linear(2880, 3072),
                nn.Sigmoid()
            )
        
        self.GAP = nn.AdaptiveAvgPool2d(1)
        
        
    def forward(self, x, mode='eval', stage=0):
        '''
        Arguments
        
        mode : choose among 'train', 'eval'. Default 'eval'
        stage : choose between 0 and 1. 0 for manifold mixing and 1 for openset-adoptation
        '''
        x = x.cuda() # bs x channel x imgsize x imgsize
        
        c = self.content_enc(x) # bs x 320, 16, 16
        mixed_c = self.mixing(c, keep_ratio=self.keep_ratio)
        s = self.style_enc(c) # bs x 640
        s_from_mc = self.style_enc(mixed_c)
        mixed_s = self.mixing(s, keep_ratio=self.keep_ratio)
        
        c = self.GAP(c).view(x.size(0),-1) # bs x 320
        mixed_c = self.GAP(mixed_c).view(x.size(0),-1) # bs x 320
        if self.backbone != 'WideResnet':
            s = self.GAP(s).view(x.size(0),-1) 
            s_from_mc = self.GAP(s_from_mc).view(x.size(0),-1)
            mixed_s = self.GAP(mixed_s).view(x.size(0),-1)
        
        id_logits = self.to_open(s)
        ood_logits = self.to_open(s_from_mc)        
        if mode=='pretrain':
            recon = self.decode(torch.cat((c, s),dim=1)) # decode의 input : bs x 960
            return id_logits, ood_logits, recon
        if mode=='train':
            recon = self.decode(torch.cat((c, mixed_s),dim=1)) # decode의 input : bs x 960
            return id_logits, ood_logits, recon
            
        else : 

            return id_logits
