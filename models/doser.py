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
    
    def mixing(self,emb, keep_ratio=0):
        idx = torch.randperm(emb.size(0))
        emb = keep_ratio * emb + (1 - keep_ratio) * emb[idx]
        return emb
    
    def to_open(self,emb):
        close_emb = self.c_classifier(emb)
        open_emb = self.o_classifier(emb)
        return torch.cat((open_emb, close_emb), dim=-1)
        
        
    def forward(self, x, mode='eval', stage=0):
        '''
        Arguments
        
        mode : choose among 'train', 'eval'. Default 'eval'
        stage : choose between 0 and 1. 0 for manifold mixing and 1 for openset-adoptation
        '''
        x = x.cuda() # bs x channel x imgsize x imgsize
        
        c = self.content_enc(x) # bs x 320, 16, 16
        mixed_c = self.mixing(c)
        s = self.style_enc(c) # bs x 640
        s_from_mc = self.style_enc(mixed_c)
        mixed_s = self.mixing(s)
        
        c = self.GAP(c).view(x.size(0),-1) # bs x 320
        mixed_c = self.GAP(mixed_c).view(x.size(0),-1) # bs x 320
        # if args.backbone != 'WideResnet':
        #     s = self.GAP(s).view(x.size(0),-1) 
        #     s_from_mc = self.GAP(s_from_mc).view(x.size(0),-1)
        #     mixed_s = self.GAP(mixed_s).view(x.size(0),-1)
        
        id_logits = self.to_open(s)
        ood_logits = self.to_open(s_from_mc)
        recon = self.decode(torch.cat((c, mixed_s),dim=1)) # decode의 input : bs x 960
        
        if mode=='train':
            return id_logits, ood_logits, recon
            
        else : 

            return id_logits.cpu()
