import time
import torch
from torch import nn
from sklearn.metrics import roc_curve, roc_auc_score

def mute_max(logits):
    min_values, _ = torch.min(logits, dim=1, keepdim=True)
    max_values, _ = torch.max(logits, dim=1, keepdim=True)
    muted = torch.where(logits == max_values, min_values, logits)
    return muted

def param_schedule(a,b,c,d,step):
    return a+step, b, c-step, d-step

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
            class_loss = class_cri(id_logits, labels)
            recon_loss = reconstruction_loss(recon.cpu(), images.view(images.size(0), -1))
            
            pre_loss = class_loss + recon_loss
            pre_loss.backward()
            pre_total_loss += pre_loss.item()
            optimizer.step()
            
        if (epoch+1) % args.print_epoch == 0:
            print(f'Pretrain epoch : {epoch}, average loss : {pre_total_loss/((idx+1)*args.print_epoch):.4f},  time elapsed : {time.time()-start_time:.4f}sec')
            start_time = time.time()
    return model

def train(args, loader, optimizer, model):
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
            ood_label = torch.ones_like(labels).to(args.device)*len(args.cls_known)
            
            optimizer.zero_grad()
            id_logits, ood_logits, recons = model(images, mode='train')
            muted_logits = mute_max(id_logits.clone())
            
            mixing_loss = class_cri(ood_logits, ood_label) 
            class_loss = class_cri(id_logits, labels) 
            muted_loss = class_cri(muted_logits, ood_label)
            if recons != None:
                recon_loss = reconstruction_loss(recons.cpu(), images.view(images.size(0), -1))
            else : # recon =None
                recon_loss = 0
            total_loss = mixing_loss * args.a + class_loss * args.b + muted_loss * args.c + recon_loss * args.d
            total_loss.backward()
            total_loss1 += mixing_loss.item()
            total_loss2 += class_loss.item()
            total_loss3 += muted_loss.item()
            total_recon += recon_loss.item()
            optimizer.step()
    
            
        if (epoch+1) % args.print_epoch == 0:
            print(f'Epoch : {epoch}, mix loss : {total_loss1/((idx+1)*args.print_epoch):.4f}, cls loss : {total_loss2/((idx+1)*args.print_epoch):.4f}, \
mute loss : {total_loss3/((idx+1)*args.print_epoch):.4f}, recon : {total_recon/((idx+1)*args.print_epoch):.4f}, \
    time elapsed : {time.time()-start_time:.4f}sec')
            start_time = time.time()

        if (epoch +1) % args.param_schedule == 0:
            args.a, args.b, args.c, args.d = param_schedule(args.a, args.b, args.c, args.d, args.param_step)    
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
    print(f'in-distribution acc: {in_acc:.5f}\nout-of-distribution acc : {ood_acc:.5f}')

    open_labels = open_labels[:n].cpu().numpy()
    prob = probs[:n].reshape(-1, 1)
    auc = roc_auc_score(open_labels, prob)
    print(f'auc : {auc:.5f}')
    return in_acc,ood_acc, auc