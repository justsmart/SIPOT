import torch
import torch.nn as nn 
# from Layers import EncoderLayer, DecoderLayer
# from Embed import Embedder, PositionalEncoder
import random
import copy
import math
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
# from model_config import model_config
from model_VAE import VAE
def gaussian_reparameterization_std(means, std, times=1):
    std = std.abs()
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times
def gaussian_reparameterization_var(means, var, times=1):
    std = torch.sqrt(var+1e-8)
    assert torch.sum(std<0).item()==0
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times
def Init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
def setEmbedingModel(d_list,d_out):
    
    return nn.ModuleList([Mlp(d,d,d_out)for d in d_list])

def fill_with_label(label_embedding,label,x_embedding,inc_V_ind):
    fea = label.matmul(label_embedding)
    new_x =  x_embedding*inc_V_ind.unsqueeze(-1) + fea.unsqueeze(1)
    return new_x

class MLP(nn.Module):
    def __init__(self, in_dim,  out_dim,hidden_dim:list=[512,1024,1024,1024,512], act =nn.GELU,norm=nn.BatchNorm1d,final_act=True,final_norm=True):
        super(MLP, self).__init__()
        self.act = act
        self.norm = norm
        # init layers
        self.mlps =[]
        layers = []
        
        if len(hidden_dim)>0:
            layers.append(nn.Linear(in_dim, hidden_dim[0]))
            # layers.append(nn.Dropout(dropout_rate))
            layers.append(self.norm(hidden_dim[0]))
            layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
            ##hidden layer
            for i in range(len(hidden_dim)-1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                # layers.append(nn.Dropout(dropout_rate))
                layers.append(self.norm(hidden_dim[i+1]))
                layers.append(self.act())
                self.mlps.append(nn.Sequential(*layers))
                layers = []
            ##output layer
            layers.append(nn.Linear(hidden_dim[-1], out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            # layers.append(nn.Dropout(dropout_rate))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
        self.mlps = nn.ModuleList(self.mlps)
    def forward(self, x):
        for layers in self.mlps:
            x = layers(x)
            # x = x + y
        return x

class Qc_inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(Qc_inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        self.z_loc = nn.Linear(out_dim, out_dim)
        self.z_sca = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())
        # self.qzv_inference = nn.Sequential(*self.qz_layer)

    def forward(self, x):
        assert torch.sum(torch.isnan(x)).item() == 0
        hidden_features = self.mlp(x)
        c_mu = self.z_loc(hidden_features)
        c_sca = self.z_sca(hidden_features)
        # class_feature  = self.z
        if torch.sum(torch.isnan(c_mu)).item() >0:
            pass
        assert torch.sum(torch.isinf(c_mu)).item() == 0
        return c_mu, c_sca

class Net(nn.Module):
    def __init__(self, d_list,num_classes,z_dim,adj,rand_seed=0):
        super(Net, self).__init__()
        # self.configs = configs
        self.rand_seed = rand_seed
        
        # Label semantic encoding module
        self.label_embedding_u = nn.Parameter(torch.eye(num_classes),
                                            requires_grad=True)
        self.label_embedding_std = nn.Parameter(torch.ones(num_classes),
                                            requires_grad=True)
        # self.label_embedding = nn.Parameter(torch.randn([num_classes,num_classes]),
        #                                     requires_grad=True)
        self.label_adj = nn.Parameter(torch.eye(num_classes),
                                      requires_grad=True)
        self.adj = adj
        self.z_dim = z_dim
        # self.GIN_encoder = GIN(2, num_classes, z_dim,
        #                        [math.ceil(z_dim / 2)])
        # self.GAT_encoder = GAT(z_dim, z_dim,dropout=0.0)
        self.label_mlp = Qc_inference_mlp(num_classes, z_dim)
        self.mix_prior = None
        self.mix_mu = None
        self.mix_sca = None
        self.k = num_classes
        # VAE module
        self.VAE = VAE(d_list=d_list,z_dim=z_dim,class_num=num_classes)
        # Classifier
        self.cls_conv = nn.Conv1d(num_classes, num_classes,
                                  z_dim*2, groups=num_classes)
        self.cls_conv2 = nn.Conv1d(num_classes, num_classes,
                                  z_dim*2, groups=num_classes)
        
        # self.cls_layers1 = MLP(z_dim,num_classes,hidden_dim=[512],final_act=False,final_norm=False)
        self.cls_layers1 = nn.Linear(z_dim,num_classes)
        self.cls_layers2 = nn.Linear(z_dim,num_classes)
        self.view_cls = nn.ModuleList([nn.Linear(z_dim, num_classes) for i in range(len(d_list)) ])
        # self.w = nn.Parameter(torch.randn([6, 260]))

        # hidden_list = [512] * (1-1)

        # self.NN3 = MLP(512, 512, hidden_list,
                    #    False, 'leaky_relu', 0.1)
        # Moving model to the right device for consistent initialization
        # self.to(configs['device'])
        # self.maxP = nn.MaxPool2d((len(d_list),1))
        self.set_prior()
        self.cuda()
        # self.reset_parameters()
    def set_prior(self):
        # set prior components weights
        self.mix_prior = nn.Parameter(torch.full((self.k,), 1 / self.k), requires_grad=True)
        # set prior gaussian components mean
        # self.mix_mu = nn.Parameter(torch.full((self.k, self.z_dim), 0.0), requires_grad=True)
        self.mix_mu = nn.Parameter(torch.rand((self.k,self.z_dim)),requires_grad=True)
        # set prior gaussian components scale
        self.mix_sca = nn.Parameter(torch.rand((self.k,self.z_dim)),requires_grad=True)
    def reset_parameters(self):
        Init_random_seed(self.rand_seed)
        # nn.init.normal_(self.label_embedding_u)
        nn.init.normal_(self.label_adj)
        # self.GIN_encoder.reset_parameters()
        self.FD_model.reset_parameters()
        self.cls_conv.reset_parameters()
        
    def get_config_optim(self):
        return [{'params': self.GAT_encoder.parameters()},
                {'params': self.FD_model.parameters()},
                {'params': self.cls_conv.parameters()}]
        
    def poe_two(self, z_mu, z_var, c_mu, c_var, eps=1e-5):
        
        z_mu = z_mu.unsqueeze(1)
        z_var = z_var.unsqueeze(1)
        c_mu = c_mu.unsqueeze(0)
        c_var = c_var.unsqueeze(0)


        # p_z_var = torch.ones([1,mu.shape[1],mu.shape[2]]).cuda()
        # mu_new = torch.cat([p_z_mu,mu],dim=0)
        # var_new = torch.cat([p_z_var,var],dim=0)
        s_mu = torch.zeros([z_mu.shape[0],c_mu.shape[1],z_mu.shape[2]]).cuda()
        s_var = torch.ones([z_mu.shape[0],c_mu.shape[1],z_mu.shape[2]]).cuda()
        # mu_new = torch.cat([p_z_mu,mu],dim=0)
        # var_new = torch.cat([p_z_var,var],dim=0)
        T_x = 1. / (z_var+eps)
        T_c = 1. / (c_var+eps)
        T_s = 1. / (s_var+eps)
        
        
        
        T_sum = T_x + T_c + T_s
        aggregate_mu = (z_mu*T_x+c_mu*T_c+ s_mu*T_s)/T_sum
        
        aggregate_var = 1. / T_sum
        # mask_matrix = torch.stack(mask, dim=0)
        
        if torch.sum(torch.isnan(aggregate_mu)).item()>0:
            print(',')
        assert torch.sum(torch.isnan(aggregate_mu)).item()==0
        assert torch.sum(torch.isinf(aggregate_mu)).item()==0
        
        assert torch.sum(torch.isnan(aggregate_var)).item()==0
        assert torch.sum(torch.isinf(aggregate_var)).item()==0
        return aggregate_mu, aggregate_var
    
    def forward(self, x_list,mask,drop_prob=0.25):
        if self.training:
            for i,X in enumerate(x_list):
                drop_len = int(drop_prob*X.size(-1))
                st = torch.randint(low=0,high=X.size(-1)-drop_len-1,size=(X.size(0),))
                # print(st,st+drop_len)
                bit_mask = torch.ones_like(X)
                for j,e in enumerate(bit_mask): 
                    bit_mask[j,st[j]:st[j]+drop_len] = 0
                x_list[i] = x_list[i].mul(bit_mask)
        # Generating semantic label embeddings via label semantic encoding module
        # label_embedding = self.GIN_encoder(self.label_embedding, self.label_adj)
        # print(self.label_adj[:10,:10])
        # label_embedding  =  self.label_embedding_u
        # label_embedding = gaussian_reparameterization_std(self.label_embedding_u,self.label_embedding_std)
        # label_embedding = self.GAT_encoder(label_embedding, self.adj)
        assert torch.sum(torch.isnan(self.label_embedding_u)).item() == 0
        assert torch.sum(torch.isinf(self.label_embedding_u)).item() == 0
        label_embedding, label_embedding_var = self.label_mlp(self.label_embedding_u)
        label_embedding_sample = gaussian_reparameterization_var(label_embedding,label_embedding_var,10)
        assert torch.sum(torch.isnan(label_embedding_sample)).item() == 0
        assert torch.sum(torch.isinf(label_embedding_sample)).item() == 0
        # label_embedding_sample = self.GAT_encoder(label_embedding_sample, self.adj)
        
        
        
        if torch.sum(torch.isnan(label_embedding)).item() > 0:
            assert torch.sum(torch.isnan(label_embedding)).item() == 0
        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list = self.VAE(x_list,mask)  #Z[i]=[128, 260, 512] b c d_e
        if torch.sum(torch.isnan(z_sample)).item() > 0:
            pass

        # uniview_emb_list = torch.stack(uniview_emb_list,dim=0)
        # for vi, uniview_emb in enumerate(uniview_emb_list):
            # uniview_emb.matmul(label_embedding_sample)
        v1 = F.normalize(z_sample, p=2, dim=1) #normalize two vectors
        v2 = F.normalize(label_embedding_sample, p=2, dim=1)
        # v2 = F.sigmoid(v2)
        sim_mat = torch.matmul(v1, v2.T) / 0.5
        sim_mat_soft = torch.softmax((sim_mat),dim=-1).detach()
        # sim_mat_soft[sim_mat_soft<0.05] = 0
        imput_sample_mu = sim_mat_soft.matmul(label_embedding)
        imput_sample_var = sim_mat_soft.matmul(label_embedding_var)
        uniview_mu = torch.stack(uniview_mu_list,dim=0) #[v n d]
        uniview_var = torch.stack(uniview_sca_list,dim=0)
        fused_uniview_mu = uniview_mu.mul(mask.T.unsqueeze(-1)) + imput_sample_mu.squeeze(dim=0).mul(1-mask.T.unsqueeze(-1))
        fused_uniview_var =  uniview_var.mul(mask.T.unsqueeze(-1)) + imput_sample_var.squeeze(dim=0).mul(1-mask.T.unsqueeze(-1))
        new_z_mu, new_z_var = self.VAE.poe_aggregate(fused_uniview_mu, fused_uniview_var, torch.ones_like(mask).cuda())
        new_z_sample = gaussian_reparameterization_var(new_z_mu, new_z_var,10)
        
        
        # mask_confi = (1-mask).mul(confi.unsqueeze(-1))+mask # b m
        # mask_confi = (1-mask).mul(confi.t())+mask # b m
        # p = p.masked_fill(mask.unsqueeze(dim=-1)==0,-1e9) 
        # p = self.maxP(p).squeeze(dim=1)
        # p = torch.max(p,dim=1)
        # print(mask_confi.shape,(mask_confi==1).sum())
        # if self.training:
        # print(mask_confi[:10,:],mask[:10,:],confi[0,:10])
        # mask_confi = mask_confi/(mask_confi.sum(dim=1,keepdim=True)+1e-9)   # b*m
        
        # p = p.mul(mask_confi.unsqueeze(dim=-1))
        # p = p.sum(dim = 1)
        # p = torch.sigmoid(qc_z)
        ###LxZ
        # xc_fusion_mu, xc_fusion_var = self.poe_two(fusion_z_mu, fusion_z_sca,label_embedding, label_embedding_var)
        # z_sample_xc = gaussian_reparameterization_var(xc_fusion_mu,xc_fusion_var)
        # p = self.cls_conv(z_sample_xc).squeeze(-1)
        ###LxZ2
        # qc_z = ((z_sample).unsqueeze(1).mul((label_embedding).unsqueeze(0)))
        # p = self.cls_conv(qc_z).squeeze(-1)


        ###LxZ3
        # label_embedding_sample = gaussian_reparameterization_var(label_embedding,label_embedding_var)
        # qc_z = ((z_sample).unsqueeze(1).mul((label_embedding_sample).unsqueeze(0)))
        # p = self.cls_conv(qc_z).squeeze(-1)
        
        ###LxZ4
        # label_embedding_sample = gaussian_reparameterization_var(label_embedding,label_embedding_var)
        # qc_z = ((z_sample).unsqueeze(1).mul((torch.sigmoid(label_embedding_sample)).unsqueeze(0)))
        # p = self.cls_conv(qc_z).squeeze(-1)
        
        ###LxZ5
        
        qc_z = torch.cat((z_sample.unsqueeze(1).repeat(1,label_embedding_sample.shape[0],1),label_embedding_sample.unsqueeze(0).repeat(z_sample.shape[0],1,1)),dim=-1)

        p = self.cls_conv(qc_z).squeeze(-1)
        # p = self.cls_layers(qc_z.view(-1,qc_z.shape[-1])).view(-1,self.k)

        ###LxZ6
        
        qc_z = torch.cat((z_sample.unsqueeze(1).repeat(1,label_embedding_sample.shape[0],1),label_embedding_sample.unsqueeze(0).repeat(z_sample.shape[0],1,1)),dim=-1)
        p = self.cls_conv(qc_z).squeeze(-1)
        new_qc_z = torch.cat((new_z_sample.unsqueeze(1).repeat(1,label_embedding_sample.shape[0],1),label_embedding_sample.unsqueeze(0).repeat(new_z_sample.shape[0],1,1)),dim=-1)
        new_p = self.cls_conv2(new_qc_z).squeeze(-1)
        # p = self.cls_layers1(z_sample)
        # new_p = self.cls_layers2(new_z_sample)
        
        
        
        ###Ori
        # p = self.cls(z_sample)
        
        # p_list = []
        # for i in range(len(x_list)):
        #     z_i = gaussian_reparameterization_var(uniview_mu_list[i],uniview_sca_list[i])
        #     z_c_i = torch.cat((z_i.unsqueeze(1).repeat(1,label_embedding_sample.shape[0],1),label_embedding_sample.unsqueeze(0).repeat(z_sample.shape[0],1,1)),dim=-1)
        #     # p_i = self.cls_conv(z_c_i).squeeze(-1)
        #     p_i = self.cls(z_c_i).squeeze(-1)
        #     p_list.append(torch.sigmoid(p_i))
        #     if torch.sum(torch.isnan(p_list[i])).item() > 0:
        #         pass
        zs_sample = gaussian_reparameterization_var(uniview_mu,uniview_var,10)
        zs_sample = torch.transpose(zs_sample, 0, 1) # [b v d]
        xr_list_cross = self.VAE.generation_x_cross(zs_sample)
        p = torch.sigmoid(p)
        new_p = torch.sigmoid(new_p)
        # x_from_label = p.matmul(label_embedding_sample)/(p.sum(dim=1,keepdim=True)+1e-8) #[n, d]
        # p_xr_list = self.VAE.generation_x_p(x_from_label)
        return z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list_cross, label_embedding_sample,p, label_embedding, label_embedding_var, new_p

def get_model(d_list,num_classes,z_dim,adj,rand_seed=0):
    
    
    model = Net(d_list,num_classes=num_classes,z_dim=z_dim,adj=adj,rand_seed=rand_seed)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() 
                                    else 'cpu'))
    return model
