import torch
from torch import nn
from .base import Basic

class Proser(Basic):
    def __init__(self,args,net):
        super().__init__(args,net)
        
    def forward(self, x, mode='eval', stage=0):
        x = x.cuda()
        if mode =='pretrain':
            c = self.encoder1(x)
            s = self.encoder2(c)
            if self.backbone !='WideResnet':
                s = self.GAP(s).view(x.size(0),-1)

            id_logits = self.to_open(s)

            return id_logits, None, None
        if mode=='train':
            # if odd, discard one
            half_len = x.size(0)//2
            g1 = x[:half_len,:,:,:].cuda()
            g2 = x[half_len:half_len*2,:,:,:].cuda()

            c1 = self.encoder1(g1)
            mixed_c1 = self.mixing(c1)
            s1 = self.encoder2(mixed_c1)
            if self.backbone !='WideResnet':
                s1 = self.GAP(s1).view(g1.size(0),-1)
            logit1 = self.to_open(s1)


            c2 = self.encoder1(g2)
            s2 = self.encoder2(c2)
            if self.backbone !='WideResnet':
                s2 = self.GAP(s2).view(g1.size(0),-1)
            logit2 = self.to_open(s2)

            return logit1, logit2, None
            
        else : 
            c = self.encoder1(x.cuda())
            s = self.encoder2(c)
            s = self.GAP(s).view(x.size(0),-1)
            logit = self.to_open(s)
            return logit


class efficient_Proser(Basic):
    def __init__(self,args,net):
        super().__init__(args,net)
        
    def forward(self, x, mode='eval', stage=0):
        x = x.cuda()
        
        c = self.encoder1(x)
        mixed_c = self.mixing(c)
        s = self.encoder2(c)
        s_from_mc = self.encoder2(mixed_c)

        
        c = self.GAP(c).view(x.size(0),-1)
        mixed_c = self.GAP(mixed_c).view(x.size(0),-1)
        if self.backbone !='WideResnet':
            s = self.GAP(s).view(x.size(0),-1)
            s_from_mc = self.GAP(s_from_mc).view(x.size(0),-1)

        
        id_logits = self.to_open(s)
        ood_logits = self.to_open(s_from_mc)
        
        if mode=='train' or mode=='pretrain':
            return id_logits, ood_logits, None
            
        else : 
            return id_logits


