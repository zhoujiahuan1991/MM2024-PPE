import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from utils import my_transform as TL
import copy
import os 
import torch.nn as nn
from torch.optim import Adam,SGD
from utils.loss import SupConLoss
from torchvision import transforms
import random
pdist = torch.nn.PairwiseDistance(p=2).cuda()
EPSILON = 1e-8



def flip_inner(x, flip1, flip2):
    #print(x.shape)

    num,channel,size,_ = x.shape#[0]
    
    # print(num)
    a = x  # .permute(0,1,3,2)
    a = a.view(num, 3, 2, size//2, size)
    #  imshow(torchvision.utils.make_grid(a))
    a = a.permute(2, 0, 1, 3, 4)
    s1 = a[0]  # .permute(1,0, 2, 3)#, 4)
    s2 = a[1]  # .permute(1,0, 2, 3)
    # print("a",a.shape,a[:63][0].shape)
    if flip1:
        s1 = torch.flip(s1, (3,))  # torch.rot90(s1, 2*rot1, (2, 3))
    if flip2:
        s2 = torch.flip(s2, (3,))  # torch.rot90(s2, 2*rot2, (2, 3))

    s = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2)
    # imshow(torchvision.utils.make_grid(s[2]))
    #   print("s",s.shape)
    # S = s.permute(0,1, 2, 3, 4)  # .view(3,32,32)
    # print("S",S.shape)
    S = s.reshape(num, 3, size, size)
    # S =S.permute(0,1,3,2)
    # imshow(torchvision.utils.make_grid(S[2]))
    #    print("S", S.shape)
    return S

def RandomFlip(x, num):
    # print(x.shape)
    #aug_x = simclr_aug(x)
   # x=simclr_aug(x)
    X = []
    # print(x.shape)

    # for i in range(4):
    X.append((x))
    X.append(flip_inner((x), 1, 1))

    X.append(flip_inner(x, 0, 1))

    X.append(flip_inner(x, 1, 0))
    # else:
    #   x1=rot_inner(x,0,1)

    return torch.cat([X[i] for i in range(num)], dim=0)

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def get_representation_matrix_ResNet18 (net, x,args=None): 
    # Collect activations by forward pass
    net.eval()
    example_data = x
    example_data = example_data.cuda()#.to(device)
    with torch.no_grad():
        example_out  = net.features(example_data)
        example_out2 = net.simclr_proj(example_out)
    act_list =[]
    act_list.extend([net.act['conv_in'], 
        net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'], net.layer1[1].act['conv_0'], net.layer1[1].act['conv_1'],
        net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1'],
        net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1'],
        net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1']])
    batch_list  =[10]*19 # [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
    # network arch 
    stride_list = [1, 1,1,1,1, 2,1,1,1, 2,1,1,1, 2,1,1,1]
    map_list    = [32, 32,32,32,32, 32,16,16,16, 16,8,8,8, 8,4,4,4] 
    #if args.nf == 20:
    #    in_channel  = [ 3, 20,20,20,20, 20,40,40,40, 40,80,80,80, 80,160,160,160] 
    #elif args.nf == 64 :
    #    in_channel  = [ 3, 64,64,64,64, 64,128,128,128, 128,256,256,256, 256,512,512,512] 
    
    in_channel  = [3]+[args.nf]*5 + [args.nf*2]*4 + [args.nf*4]*4 + [args.nf*8]*3
    pad = 1
    sc_list=[5,9,13]
    p1d = (1, 1, 1, 1)
    mat_final=[] # list containing GPM Matrices 
    mat_list=[]
    mat_sc_list=[]
    for i in range(len(stride_list)):
        if i==0:
            ksz = 3
        else:
            ksz = 3 
        bsz=batch_list[i]
        st = stride_list[i]     
        k=0
        s=compute_conv_output_size(map_list[i],ksz,stride_list[i],pad)
        mat = np.zeros((ksz*ksz*in_channel[i],s*s*bsz))
        act = F.pad(act_list[i], p1d, "constant", 0).detach().cpu().numpy()
        for kk in range(bsz):
            for ii in range(s):
                for jj in range(s):
                    mat[:,k]=act[kk,:,st*ii:ksz+st*ii,st*jj:ksz+st*jj].reshape(-1)
                    k +=1
        mat_list.append(mat)
        # For Shortcut Connection
        if i in sc_list:
            k=0
            s=compute_conv_output_size(map_list[i],1,stride_list[i])
            mat = np.zeros((1*1*in_channel[i],s*s*bsz))
            act = act_list[i].detach().cpu().numpy()
            for kk in range(bsz):
                for ii in range(s):
                    for jj in range(s):
                        mat[:,k]=act[kk,:,st*ii:1+st*ii,st*jj:1+st*jj].reshape(-1)
                        k +=1
            mat_sc_list.append(mat) 
    
    if args.proj_gpm :

        bsz = batch_list[17]
        act = net.act['proj'].detach().cpu().numpy()
        activation = act[0:bsz].transpose()
        mat_list.append(activation)
        
    ik=0
    for i in range (len(mat_list)):
        mat_final.append(mat_list[i])
        if i in [6,10,14]:
            mat_final.append(mat_sc_list[ik])
            ik+=1
    
    print('-'*30)
    print('Representation Matrix')
    print('-'*30)
    for i in range(len(mat_final)):
        1
        #print ('Layer {} : {}'.format(i+1,mat_final[i].shape))
    print('-'*30)
    return mat_final    


def update_GPM (model, mat_list, threshold, feature_list=[],):

    if not feature_list:
        # After First Task 
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U,S,Vh = np.linalg.svd(activation, full_matrices=False)

            sval_total = (S**2).sum()
            sval_ratio = (S**2)/sval_total
            r = np.sum(np.cumsum(sval_ratio)<threshold[i]) #+1  
            feature_list.append(U[:,0:r])
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            try :
                U1,S1,Vh1=np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                # Projected Representation (Eq-8)
                act_hat = activation - np.dot(np.dot(feature_list[i],feature_list[i].transpose()),activation)
                U,S,Vh = np.linalg.svd(act_hat, full_matrices=False)
                # criteria (Eq-9)
                sval_hat = (S**2).sum()
                sval_ratio = (S**2)/sval_total               
                accumulated_sval = (sval_total-sval_hat)/sval_total
            except Exception as e :
                print('svd not converage')
            r = 0
            for ii in range (sval_ratio.shape[0]):
                if accumulated_sval < threshold[i]:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break
            if r == 0:
                print ('Skip Updating GPM for layer: {}'.format(i+1)) 
                continue
            # update GPM
            Ui=np.hstack((feature_list[i],U[:,0:r]))  
            if Ui.shape[1] > Ui.shape[0] :
                feature_list[i]=Ui[:,0:Ui.shape[0]]
            else:
                feature_list[i]=Ui
    
    print('-'*40)
    print('Gradient Constraints Summary')
    print('-'*40)
    for i in range(len(feature_list)):
        print ('Layer {} : {}/{}'.format(i+1,feature_list[i].shape[1], feature_list[i].shape[0]))
    print('-'*40)
    return feature_list  

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def dis (x,y):
    d = torch.cdist(x,y)

    return d

class Prototypes(nn.Module):
    def __init__(
        self,
        n_classes,
        feat_dim: int, init,args,
    ):
        super(Prototypes, self).__init__()
        self.args= args

        self.index = [0 for i in range(n_classes)]
        num = self.args.proto_num
        self.num = num
        if init == 'randn' :
            self.prototype = nn.Parameter(torch.randn(num,n_classes,feat_dim),requires_grad = True)#.cuda()
        elif init == 'zeros' :
            self.prototype = nn.Parameter(torch.zeros(num,n_classes,feat_dim),requires_grad = True)#.cuda()
        
        #self.prototype.
        
    def loss(self, feature, y,label,y_pred=None,batch_index=1000) -> torch.FloatTensor:
        p_loss = 0
        num = 0
        
        feature_ = nn.functional.normalize(feature, dim=-1)
        feature_ = feature_.detach()
        feature_.requires_grad = False


        feature_ = feature_.view(self.args.proto_num,-1,feature_.shape[-1])
        y = y.view(self.args.proto_num,-1)

        if y_pred != None :
            entropys = softmax_entropy(y_pred)
            filter_ids_1 =entropys < 1000000
            filter_ids_1 = filter_ids_1.view(self.args.proto_num,-1)

        unique_labels = torch.unique(y)
        for c in unique_labels:
            if c.item() not in label:
                label.append(c.item())
        for i in range(self.num):
            filter_n = filter_ids_1[i]
            y_n = y[i][filter_n]
            feature_n = feature_[i][filter_n]
            k = nn.functional.normalize(self.prototype[i,y_n], dim=-1)

            dis = torch.pow(feature_n-k,2)
            dis = torch.sum(dis,dim=-1)
            dis = torch.sqrt(dis)
            p_loss = dis.sum()+ p_loss
            num += feature_n.shape[0]

        return p_loss/num
    def update(self, feature, y) -> torch.FloatTensor:

        feature = feature.detach()
        feature.requires_grad = False
        feature = feature[:self.args.batch_size]
        y = y[:self.args.batch_size]
        #print(self.prototype[:,:10].max(),self.prototype[:,:10].min())
        for i in range(len(y)):
            label = y[i]
            this_index = self.index[label]
            self.index[label] = (this_index+1)%self.num 
            self.prototype[this_index][label] = feature[i]
            

class TrainLearnerGPM(object):
    def __init__(self, model, optimizer, n_classes_num, class_per_task, input_size, args, fea_dim=128):
        self.args = args
        self.model = model
        self.model.prototype = Prototypes(n_classes_num,args.nf*8*4,'randn',args=args) .cuda()

        self.ema_model =copy.deepcopy( model).cuda()
        self.old_model =copy.deepcopy( model).cuda()
        self.optimizer = optimizer
        self.n_classes_num = n_classes_num
        self.fea_dim = fea_dim
        self.classes_mean = torch.zeros((n_classes_num, fea_dim), requires_grad=False).cuda()
        self.class_per_task = class_per_task
        self.class_holder = []




        self.dataset = args.dataset
        if args.dataset == "cifar10":
            self.sim_lambda = 0.5
            self.total_samples = 10000
        elif "cifar100" in args.dataset:
            self.sim_lambda = 1.0
            self.total_samples = 5000
        elif args.dataset == "miniimagenet":
            self.sim_lambda = 1.0
            self.total_samples = 5000
        self.print_num = self.total_samples // 10
        self.CON = SupConLoss(temperature=0.09, contrast_mode='all')


        self.prototype_label = []

        with torch.no_grad():
            resize_scale = (0.6, 1.0)  # resize scaling factor,default [0.08,1]

            color_gray = TL.RandomColorGrayLayer(p=0.2).cuda()
            resize_crop = TL.RandomResizedCropLayer(scale=resize_scale, size=[input_size[1], input_size[2], input_size[0]]).cuda()
            self.simclr_aug = torch.nn.Sequential(color_gray, resize_crop,
                )

        self.threlist = []

        
    def train_task0(self, task_id, train_loader):

        num_d = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            batchsize = x.shape[0]
            num_d += x.shape[0]

            str_loss = 0
            proto_loss = 0
            con_loss = 0
            ce2 = 0
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            x = x.requires_grad_()

            x_1 = RandomFlip(self.simclr_aug(x), 2)
            x_2 = torch.cat([self.simclr_aug(x)  for _ in range(2)])
            x = torch.cat((x_1,x_2))
            y = y.repeat(4)

            feature = self.model.features(x)
            
            if self.args.proj_gpm :
                proj = self.model.simclr_proj(feature)
            else :
                proj = feature
            f1, f2 = torch.split(proj[:batchsize*2,:], [batchsize, batchsize], dim=0)
            #f1 = torch.cat([f1,proto_features],dim=0)
            #f2 = torch.cat([f2,proto_features],dim=0)
            f1_norm = torch.norm(f1, p=2, dim=1).unsqueeze(1).expand_as(f1)
            f1 = f1.div(f1_norm + 0.000001)
            f2_norm = torch.norm(f2, p=2, dim=1).unsqueeze(1).expand_as(f2)
            f2 = f2.div(f2_norm + 0.000001)
            y_con, _ = torch.split(y[:batchsize*2], [batchsize, batchsize], dim=0)
            #y_con= torch.cat([y_con,proto_targets])

            fff = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            con_loss = self.CON(fff,y_con)*self.args.weight_con
            if batch_idx <= self.args.con_begin :
                con_loss = 0
            
            
                
            y_pred = self.model.logit(feature)[0]

            proto_loss = self.model.prototype.loss(feature,y,self.prototype_label,y_pred,batch_idx)

            if batch_idx >= self.args.proto_ce :
                begin = task_id*self.class_per_task
                end = (task_id+1)*self.class_per_task
                proto_features =copy.deepcopy( self.model.prototype.prototype[0,begin:end].detach() )
                proto_features = proto_features.reshape(-1,proto_features.shape[-1])
                proto_targets = np.arange(self.class_per_task) #np.array([i for i in range(self.class_per_task)])
                proto_targets = torch.from_numpy(proto_targets).cuda()
                #proto_targets = proto_targets.repeat(self.model.prototype.num)
                proto_targets = proto_targets.repeat(1)
                


                #y= torch.cat([y,proto_targets])
                proto_pred = self.model.logit(proto_features)[0]
                
                #y_pred = torch.cat([y_pred,proto_pred])
            if batch_idx >= self.args.proto_ce :

                ce1 = F.cross_entropy(y_pred, y)
                ce2 = F.cross_entropy(proto_pred, proto_targets) * self.args.gamma 
            else : 
                ce1 = F.cross_entropy(y_pred, y) 

            ce = ce1 + ce2


                
            #self.model.prototype_save.update(feature,y)
            #print(self.model.logit(feature).shape)
            
            #print(torch.max(y_pred.softmax(1),dim=-1).values)
            #print('entropy',entropy)
            #print(y.max(),y.min())
            
            self.optimizer.zero_grad()
            loss = ce + str_loss + con_loss + proto_loss

            
            loss.backward()

            self.optimizer.step()

            self.ema_update(task_id)
            self.lr_update(batch_idx)

            if num_d % self.print_num == 0 or batch_idx == 1:
                print(
                    '==>>> it: {}, loss: ce {:.2f} + str {:.4f} + con {:.4f} + PR {:.4f}  = {:.6f}, {}%'
                    .format(batch_idx, ce, str_loss,con_loss,proto_loss ,loss, 100 * (num_d / self.total_samples)))
        print('total train num: {}'.format(num_d))
        return x 
    
    def train_other_tasks(self, task_id, train_loader,feature_mat):
        num_d = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            num_d += x.shape[0]
            batchsize = x.shape[0]

            str_loss = 0
            proto_loss = 0
            con_loss = 0
            ce2 = 0
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            x = x.requires_grad_()


            x_1 = RandomFlip(self.simclr_aug(x), 2)
            x_2 = torch.cat([self.simclr_aug(x)  for _ in range(2)])
            x = torch.cat((x_1,x_2))
            y = y.repeat(4)

            feature = self.model.features(x)

            index = np.arange(self._known_classes)
            proto_features =copy.deepcopy( self.model.prototype.prototype[:,:self._known_classes].detach() )
            proto_features = proto_features.reshape(-1,proto_features.shape[-1])
            proto_targets = index
            proto_targets = torch.from_numpy(proto_targets).cuda()
            if  self.args.proj_gpm :
                proj = self.model.simclr_proj(feature)
                proto_features_proj = self.model.simclr_proj(proto_features)
            else :
                proj = feature
                proto_features_proj = proto_features
            y_con, _ = torch.split(y[:batchsize*2], [batchsize, batchsize], dim=0)
            f1, f2 = torch.split(proj[:batchsize*2,:], [batchsize, batchsize], dim=0)
            
            f1_norm = torch.norm(f1, p=2, dim=1).unsqueeze(1).expand_as(f1)
            f1 = f1.div(f1_norm + 0.000001)
            f2_norm = torch.norm(f2, p=2, dim=1).unsqueeze(1).expand_as(f2)
            f2 = f2.div(f2_norm + 0.000001)
            
            fff = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            con_loss = self.CON(fff,y_con)*self.args.weight_con
            if batch_idx <= self.args.con_begin :
                con_loss = 0

                    

            feature_old = self.ema_model.features(x)

            proj_old = feature_old
            proj = feature
            proto_features_proj = proto_features
            proj_norm = torch.norm(proj, p=2, dim=1).unsqueeze(1).expand_as(proj)
            proj_normalized = proj.div(proj_norm + 0.000001)
            proj_old_norm = torch.norm(proj_old, p=2, dim=1).unsqueeze(1).expand_as(proj_old)
            proj_old_normalized = proj_old.div(proj_old_norm + 0.000001)
            proto_features_proj_norm = torch.norm(proto_features_proj, p=2, dim=1).unsqueeze(1).expand_as(proto_features_proj)
            proto_features_proj_normalied = proto_features_proj.div(proto_features_proj_norm + 0.000001)
            f_p_sim = torch.matmul(proj_normalized, proto_features_proj_normalied.T)
            fo_p_sim = torch.matmul(proj_old_normalized, proto_features_proj_normalied.T)

            str_loss = torch.mean(abs(f_p_sim-fo_p_sim)) * self.args.miu

            

            
            y_pred = self.model.logit(feature)[task_id]
               
            proto_loss = self.model.prototype.loss(feature,y,self.prototype_label,y_pred,batch_idx)      

            if batch_idx >= self.args.proto_ce :
                begin = task_id*self.class_per_task
                end = (task_id+1)*self.class_per_task
                proto_features =copy.deepcopy( self.model.prototype.prototype[0,begin:end].detach() )
                proto_features = proto_features.reshape(-1,proto_features.shape[-1])
                proto_targets = np.array([i for i in range(begin,end)])
                proto_targets = torch.from_numpy(proto_targets).cuda()

                proto_targets = proto_targets.repeat(1)
                proto_pred = self.model.logit(proto_features)[task_id]

            if batch_idx >= self.args.proto_ce :

                ce1 = F.cross_entropy(y_pred, y-task_id*self.class_per_task)
                ce2 = F.cross_entropy(proto_pred, proto_targets-task_id*self.class_per_task) * self.args.gamma 
            else : 
                ce1 = F.cross_entropy(y_pred, y-task_id*self.class_per_task) 
            ce = ce1+ce2


            loss = ce + str_loss + con_loss + proto_loss
            self.optimizer.zero_grad()

            loss.backward()
            kk = 0 
            for k, (m,params) in enumerate(self.model.named_parameters()):

                if 'simclr'  in m :
                    if self.args.proj_gpm and params.grad!=None:
                        sz =  params.grad.data.size(0)
                        params.grad.data = params.grad.data -  torch.mm(params.grad.data.view(sz,-1),\
                                                                feature_mat[kk]).view(params.size())
                        #print(kk)
                        kk += 1
                    else:
                        continue
                if len(params.size())==4 :
                    #print(1)
                    #print(m,params.size(),kk)
                    sz =  params.grad.data.size(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz,-1),\
                                                            feature_mat[kk]).view(params.size())
                    kk+=1
                elif len(params.size())==1 and task_id !=0:
                    #print(k,m,params.shape)
                    params.grad.data.fill_(0)
            self.optimizer.step()

            self.ema_update(task_id)
            
            self.lr_update(batch_idx)

            if num_d % self.print_num == 0 or batch_idx == 1:
                print(
                    '==>>> it: {}, loss: ce {:.2f} + str {:.4f} + con {:.4f} + PR {:.4f}  = {:.6f}, {}%'
                    .format(batch_idx, ce, str_loss,con_loss,proto_loss ,loss, 100 * (num_d / self.total_samples)))
        print('total train num: {}'.format(num_d))
        return x

    def lr_update(self,batch_idx) :
        if batch_idx == 399 or batch_idx ==449 :
            for param_group in self.optimizer.param_groups:
                if param_group['name'] == 'prototype' :
                    param_group['lr'] /= self.args.lr_factor  


    
    def ema_update(self,task_id):
        ema_model_dict = self.ema_model.state_dict() 
        model_dict = self.model.state_dict() 
        for par1, par2 in zip(ema_model_dict,model_dict):
            ema_model_dict[par1] =     0.99*ema_model_dict[par1] + (1-0.99)*model_dict[par2] 
        self.ema_model.load_state_dict(ema_model_dict)


    def old_copy(self):
        old_model_dict = self.old_model.state_dict() 
        model_dict = self.model.state_dict() 
        for par1, par2 in zip(old_model_dict,model_dict):
            old_model_dict[par1] =   model_dict[par2] 
        self.old_model.load_state_dict(old_model_dict)        
    
    def train(self, task_id, train_loader,task_Test_loader=None):
   
        path = os.path.join('logs',self.args.log_dir)
        for epoch in range(1):
            self.model.train()
            self.ema_model.train()
            self.model.prototype.train()

            p = []
            for n,m in self.model.named_parameters():
                if 'prototype' not in n :
                    p.append(m)

            x = [{  "params": p,
                    "lr": self.args.lr,'name':'model',}]
            x .append({
                "params": self.model.prototype.prototype,
                "lr": self.args.prototypes_lr,
                'name':'prototype',
            })

            if self.args.optimi == 'SGD':
                
                
                self.optimizer = SGD(params=x)
            elif self.args.optimi == 'Adam':
                self.optimizer = Adam(x, betas=(0.9, 0.99), weight_decay=1e-4)

            if task_id == 0:
                self.feature_list =[]
                
                x_train = self.train_task0(task_id, train_loader)
            
            else:
                #self.feature_list = []
                feature_mat = []
                self.feature_list_len = [ i.shape[1] for i in self.feature_list]
                # Projection Matrix Precomputation
                for i in range(len(self.feature_list)):
                    Uf=torch.Tensor(np.dot(self.feature_list[i],self.feature_list[i].transpose())).cuda()#.to(device)
                    #print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
                    feature_mat.append(Uf)
                print ('-'*40)

                x_train = self.train_other_tasks(task_id, train_loader,feature_mat)
        
        
        if not os.path.isdir(path):
            os.makedirs(path)
        state = {'net':self.model.state_dict()}
        torch.save(state, "{}.pkl".format(os.path.join(path,str(task_id)))) 

        self.model.eval()
        self.old_copy()


        if task_id>0 :
            f1 = self.model.prototype.prototype[0,(task_id-1)*self.class_per_task:(task_id)*self.class_per_task,:]
            f2 = self.model.prototype.prototype[0,(task_id)*self.class_per_task:(task_id+1)*self.class_per_task,:]

            f1 = nn.functional.normalize(f1.view(-1,f1.shape[-1]), dim=-1)
            f2 = nn.functional.normalize(f2.view(-1,f2.shape[-1]), dim=-1).T
        
            sim = torch.matmul(f1,f2)
            thre = self.args.alpha-self.args.beta*torch.mean(sim).data
            self.threlist.append(thre.detach().cpu().numpy().tolist())
            threshold = np.array([thre.detach().cpu().numpy()] * 20 + [0.965,0.965]) 
        else :
            self.threlist.append(self.args.threshold)
            threshold = np.array([self.args.threshold] * 20 + [0.965,0.965]) 
        
        mat_list = get_representation_matrix_ResNet18 (self.model,  x_train, args=self.args)
        self.feature_list = update_GPM (self.model, mat_list, threshold, self.feature_list)

        self._known_classes =(task_id+1)*self.class_per_task #len(self.prototype_label)
        print('-----------thre list--------------------')
        print(self.threlist)
    
    def test(self, i, task_loader,task_id):
        
        self.model.eval()
        self.ema_model.eval()
        self.model.prototype.eval()
        r1 = []
        r2 = []
        for name in ['Linear','NME'] :
            print('-----------{} ACC----------'.format(name))
            l = (i+1)*self.class_per_task
            if name == 'Linear' or (name=='NME' and self.args.test_nme) :
                acc_matrix = torch.zeros((l,l)).cuda()
                acc_matrix_ema= torch.zeros((l,l)).cuda()
            with torch.no_grad():
                acc_list = np.zeros(len(task_loader))
                acc_list_norm = np.zeros(len(task_loader))
                acc_ema_list = np.zeros(len(task_loader))
                acc_ema_list_norm = np.zeros(len(task_loader))

                for j in range(i + 1):
                    #loader = 
                    data_all = []
                    for batch_idx, (data, target) in enumerate(task_loader[j]['test']):
                        data_all.append([data,target])
                    acc,results = self.test_model(self.model,data_all, j,i,name=name)
                    if name == 'NME':
                        acc_norm,results_norm = self.test_model(self.model,data_all, j,i,True,name=name)
                    else :
                        acc_norm,results_norm =acc,results
                    acc_ema,results_ema= self.test_model(self.ema_model,data_all, j,i,name=name)
                    if name == 'NME':
                        acc_norm_ema,results_norm_ema = self.test_model(self.ema_model,data_all, j,i,True,name=name)
                    else :
                        acc_norm_ema,results_norm_ema =acc_ema,results_ema
                    acc_ema_list[j] = acc_ema.item()
                    acc_list[j] = acc.item()
                    acc_ema_list_norm[j] = acc_norm_ema.item()
                    acc_list_norm[j] = acc_norm.item()
                    if name == 'Linear' or (name=='NME' and self.args.test_nme) :
                        #print(results_norm)
                        acc_matrix[j*self.class_per_task:(j+1)*self.class_per_task]=results_norm
                        acc_matrix_ema[j*self.class_per_task:(j+1)*self.class_per_task]=results_norm_ema

                print(f"tasks acc:{acc_list_norm}")
                print(f"tasks avg acc:{acc_list_norm[:i+1].mean()}")
                print(f"tasks ema acc:{acc_ema_list_norm}")
                print(f"tasks ema avg acc:{acc_ema_list_norm[:i+1].mean()}")

                r1.append(acc_list_norm)
                r2.append(acc_ema_list_norm)

        return r1[0],r2[0], r1[1],r2[1]

    def test_model(self,model, loader, i,task_id,norm=False,name=None):
        
        result = torch.zeros((self.class_per_task,(task_id+1)*self.class_per_task)).cuda()
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            target_ = target
            
            if name == 'NME' :
                pred_feature = model.features(data)
                pro = model.prototype.prototype[0]

                if norm :
                    pred_feature = pred_feature/(torch.norm(pred_feature,dim=-1,keepdim=True)+EPSILON)#.unsqueeze(1)
                    prototype = pro/(torch.norm(pro,dim=-1,keepdim=True)+EPSILON)#.unsqueeze(1)
                else:
                    prototype = pro

                sim = torch.cdist(pred_feature.unsqueeze(0),prototype.unsqueeze(0))[0][:,:self._known_classes]
                Pred = sim.argmin(dim=1, keepdim=True) 
                
            elif name == 'Linear' :
                pred = model(data)
                pred = torch.cat(pred[:task_id+1],dim=-1)
                Pred = pred.data.max(1, keepdim=True)[1]

            ids = Pred[:,0]#//self.class_per_task
            for jjj in range(len(ids)) :
                result[target[jjj]- i * self.class_per_task][ids[jjj]] += 1 
            num += data.size()[0]
            correct += Pred.eq(target_.data.view_as(Pred)).sum()

        test_accuracy = (100. * correct / num)
        return test_accuracy,result

    def test_model_ema(self, loader, i,task_id,norm=False):
        result = torch.zeros((self.class_per_task,(task_id+1)*self.class_per_task)).cuda()
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            target_ = target
            if self.args.test_nme :
                pred_feature = self.ema_model.features(data)
                pro = self.model.prototype.prototype[0]

                if norm :
                    pred_feature = pred_feature/(torch.norm(pred_feature,dim=-1,keepdim=True)+EPSILON)#.unsqueeze(1)
                    prototype = pro/(torch.norm(pro,dim=-1,keepdim=True)+EPSILON)#.unsqueeze(1)
                else:
                    prototype = pro
                sim = torch.cdist(pred_feature.unsqueeze(0),prototype.unsqueeze(0))[0][:,:self._known_classes]
                Pred = sim.argmin(dim=1, keepdim=True) 
            else :
                pred = self.ema_model(data)
                pred = torch.cat(pred[:task_id+1],dim=-1)
                Pred = pred.data.max(1, keepdim=True)[1]
            
            ids = Pred[:,0]#//self.class_per_task
            for jjj in range(len(ids)) :
                result[target[jjj]- i * self.class_per_task][ids[jjj]] += 1 

            num += data.size()[0]
            correct += Pred.eq(target_.data.view_as(Pred)).sum()

        test_accuracy = (100. * correct / num)

        return test_accuracy,result


