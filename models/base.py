import torch
from torch import nn
import torch.nn.functional as F

class pre2block(nn.Module):
    def __init__(self,net):
        super(pre2block, self).__init__()
        self.net = net
    def forward(self,x):
        out = self.net.conv1(x)
        out = self.net.layer1(out)
        out = self.net.layer2(out)
        return out

class latter2block(nn.Module):
    def __init__(self,net):
        super(latter2block, self).__init__()
        self.net = net    
    def forward(self,x):
        out = self.net.layer3(x)
        out = F.relu(self.net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return out # bs x 640

class Basic(nn.Module):
    def __init__(self,args,net):
        super(Basic, self).__init__()

        self.net = net
        self.encoder1 =pre2block(net)
        self.encoder2 =latter2block(net)
        self.num_class = len(args.cls_known)
        # closed set classifier
        self.c_classifier =  nn.Sequential(
            nn.Linear(640, 320),
            nn.ReLU(),
            nn.Linear(320,80),
            nn.ReLU(),
            nn.Linear(80,self.num_class)
        )
        
        # open set classifier
        self.o_classifier = nn.Sequential(
            nn.Linear(640, 320),
            nn.ReLU(),
            nn.Linear(320,160),
            nn.ReLU(),
            nn.Linear(160,40),
            nn.ReLU(),
            nn.Linear(40,1)
        )
        
        
        self.GAP = nn.AdaptiveAvgPool2d(1)
    
    def mixing(self,emb, keep_ratio=0):
        idx = torch.randperm(emb.size(0))
        emb = keep_ratio * emb + (1 - keep_ratio) * emb[idx]
        return emb
    
    def to_open(self,emb):
        close_emb = self.c_classifier(emb)
        open_emb = self.o_classifier(emb)
        return torch.cat((close_emb, open_emb), dim=-1)
        
        
    def forward(self, x, mode='eval', stage=0):
        '''
        Arguments
        
        mode : choose among 'train', 'eval'. Default 'eval'
        stage : choose between 0 and 1. 0 for manifold mixing and 1 for openset-adoptation
        '''
        pass