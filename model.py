import numpy as np
import torch
from PIL import Image
from torchvision.datasets.mnist import read_image_file, read_label_file
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from torchvision import transforms, utils, models
import torch.nn as nn

class openset_convnet(nn.Module):
    def __init__(self):
        super(openset_convnet, self).__init__()
        
        # Encoder before manifold mix up
        self.content_enc = nn.Sequential(
            #
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            #
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        
        self.style_enc  = nn.Sequential(
            #
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            #
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # closed set classifier
        self.c_classifier =  nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,9)
        )
        
        # open set classifier
        self.o_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
        
        self.decode= nn.Sequential(
            nn.Linear(32+64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
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
        x = x.cuda()
        
        c = self.content_enc(x)
        mixed_c = self.mixing(c)
        s = self.style_enc(c)
        s_from_mc = self.style_enc(mixed_c)
        mixed_s = self.mixing(s)
        
        c = self.GAP(c).view(x.size(0),-1)
        mixed_c = self.GAP(mixed_c).view(x.size(0),-1)
        s = self.GAP(s).view(x.size(0),-1)
        s_from_mc = self.GAP(s_from_mc).view(x.size(0),-1)
        mixed_s = self.GAP(mixed_s).view(x.size(0),-1)
        
        id_logits = self.to_open(s)
        ood_logits = self.to_open(s_from_mc)
        recon = self.decode(torch.cat((c, mixed_s),dim=1))
        
        if mode=='train':
            return id_logits, ood_logits, recon
            
        else : 

            return id_logits.cpu()
