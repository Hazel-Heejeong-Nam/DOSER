import time
import torch
from torch import nn
from sklearn.metrics import roc_curve, roc_auc_score
from utils import mute_max, param_schedule
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def pretrain(args, loader, optimizer, model):
    class_cri = nn.CrossEntropyLoss()
    reconstruction_loss = nn.MSELoss()
    start_time = time.time()
    for epoch in range(args.pretrain_epochs):
        pre_total_loss = 0
        for idx, (images, labels) in enumerate(loader):
    
            optimizer.zero_grad()
            labels = labels.to(args.device) 
            
            id_logits, _, recon = model(images,mode='pretrain')
            class_loss = class_cri(id_logits, labels[:len(id_logits)])
            if recon==None :
                recon_loss = 0
            else :
                recon_loss = reconstruction_loss(recon.cpu(), images.view(images.size(0), -1))
            
            pre_loss = class_loss + recon_loss
            pre_loss.backward()
            pre_total_loss += pre_loss.item()
            optimizer.step()
            
        if (epoch+1) % args.print_epoch == 0:
            print(f'Pretrain epoch : {epoch}, average loss : {pre_total_loss/((idx+1)*args.print_epoch):.4f},  time elapsed : {time.time()-start_time:.4f}sec')
            start_time = time.time()
    return model

def train(args, loader, optimizer, model, scheduler, params):
    a,b,c,d =params
    #print(a,b,c,d)
    class_cri = nn.CrossEntropyLoss()
    reconstruction_loss = nn.MSELoss()
    # Training
    start_time = time.time()
    for epoch in range(args.train_epochs):
        total_loss1 = 0
        total_loss2 = 0
        total_loss3 = 0
        total_recon = 0
        for idx, (images, labels) in enumerate(loader):
            
            labels = labels.to(args.device) # labels 32  
            half_len = labels.size(0)//2
            ood_label = torch.ones_like(labels).to(args.device)*len(args.cls_known)
            
            optimizer.zero_grad()
            id_logits, ood_logits, recons = model(images, mode='train')
            muted_logits = mute_max(id_logits.clone())
            
            mixing_loss = class_cri(ood_logits, ood_label[:len(id_logits)]) 
            if args.model=='Proser':
                class_loss = class_cri(id_logits, labels[half_len:half_len*2]) 
            else :
                class_loss = class_cri(id_logits, labels) 
            muted_loss = class_cri(muted_logits, ood_label[:len(id_logits)])
            if recons != None:
                recon_loss = reconstruction_loss(recons.cpu(), images.view(images.size(0), -1))
            else : # recon =None
                recon_loss = 0
            total_loss = mixing_loss * a + class_loss * b + muted_loss * c + recon_loss * d
            total_loss.backward()
            total_loss1 += mixing_loss.item()
            total_loss2 += class_loss.item()
            total_loss3 += muted_loss.item()
            if recons != None : 
                total_recon += recon_loss.item()
            optimizer.step()
    
            
        if (epoch+1) % args.print_epoch == 0:
            if recons !=None : 
                print(f'Epoch : {epoch}, mix loss : {total_loss1/((idx+1)*args.print_epoch):.4f}, cls loss : {total_loss2/((idx+1)*args.print_epoch):.4f}, \
mute loss : {total_loss3/((idx+1)*args.print_epoch):.4f}, recon : {total_recon/((idx+1)*args.print_epoch):.4f}, \
    time elapsed : {time.time()-start_time:.4f}sec')
            else :
                print(f'Epoch : {epoch}, mix loss : {total_loss1/((idx+1)*args.print_epoch):.4f}, cls loss : {total_loss2/((idx+1)*args.print_epoch):.4f}, \
mute loss : {total_loss3/((idx+1)*args.print_epoch):.4f}, time elapsed : {time.time()-start_time:.4f}sec')


            start_time = time.time()

        if (epoch +1) % args.param_schedule == 0:
            a,b,c,d = param_schedule(a,b,c,d, args.param_step)   
        if args.lr_schedule ==True:
            scheduler.step() 
    return model
            
def validate(args,model, testloader, outloader):  
    model.eval()
    correct_id, total_id, correct_ood, total_ood, n = 0, 0, 0, 0, 0
    torch.cuda.empty_cache()
    open_labels = torch.zeros(50000)
    probs = torch.zeros(50000)
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.cuda(), labels.cuda()
            bs = labels.size(0)
            logits = model(data) # bs = (cls_known +1)
            total_logits = torch.softmax(logits, dim=1)
            id_logits = torch.softmax(logits[:,:len(args.cls_known)], dim=1)
            confidence = id_logits.data.max(1)[0] # in-distribution 중에서 젤 큰 prob [64]
            for b in range(bs):
                probs[n] = confidence[b]
                open_labels[n] = 1 # in-distribution 중에서 젤 큰 prob이 1에 가까워야하니까
                n += 1
            predictions = total_logits.data.max(1)[1] # ood 포함 전체 라벨 뭘로 pred 했는지
            total_id += labels.size(0)
            correct_id += (predictions == labels.data).sum()


        for data, labels in outloader:
            data, labels = data.cuda(), labels.cuda()
            oodlabel = torch.ones_like(labels).cuda()*len(args.cls_known)
            bs = labels.size(0)
            logits = model(data)
            total_logits = torch.softmax(logits, dim=1)
            id_logits = torch.softmax(logits[:,:len(args.cls_known)], dim=1)
            confidence = id_logits.data.max(1)[0] # in-distribution 중에서 젤 큰 prob
            for b in range(bs):
                probs[n] = confidence[b] # 0에 가까울수록 잘한거임
                open_labels[n] = 0 # in-distribution 중에서 젤 큰 prob이 0에 가까워야하니까
                n += 1
            predictions = total_logits.data.max(1)[1] # ood 포함 전체 라벨 뭘로 pred 했는지
            total_ood += labels.size(0)
            correct_ood += (predictions == oodlabel).sum()

    in_acc = float(correct_id) * 100. / float(total_id)
    ood_acc = float(correct_ood) * 100. / float(total_ood)
    avg_acc = float(correct_id+correct_ood)* 100. / float(total_id+total_ood)
    print(f'in-distribution acc: {in_acc:.5f}\nout-of-distribution acc : {ood_acc:.5f}\n average acc : {avg_acc:.5f}')

    open_labels = open_labels[:n].cpu().numpy()
    prob = probs[:n].reshape(-1, 1)
    auc = roc_auc_score(open_labels, prob)
    print(f'auc : {auc:.5f}')
    return in_acc,ood_acc,avg_acc, auc

def get_pic(args, model, train_loader):
    model.eval()
    for idx, (images, labels) in enumerate(train_loader):
        id_logits, ood_logits, fake = model(images, mode='analysis')
        fake = fake.detach().cpu().reshape(images.shape).permute(0,2,3,1).numpy()
        for i in range(10):
            img = fake[i]
            plt.imsave(f'{args.model}_{i}.jpg', img)
        if idx ==0 :
            break

