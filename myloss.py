import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
def gaussian_reparameterization_var(means, var, times=1):
    std = torch.sqrt(var)
    res = torch.zeros_like(means).to(means.device)
    for t in range(times):
        epi = std.data.new(std.size()).normal_()
        res += epi * std + means
    return res/times
def W_distance(p_mu,p_v,q_mu,q_v,eps=1e-9):
    p_var = p_v+eps
    q_var = q_v+eps
    res = torch.sum((p_mu-q_mu)**2,dim=-1)+p_var.sum(-1)+q_var.sum(-1)- 2*((p_var*q_var)**0.5).sum(-1)
    return res
def gaussian_log_var(x, mu, var, eps=1e-14):
    return -0.5 * (torch.log(torch.tensor(2.0 * np.pi)) + torch.log(var) + torch.pow(x - mu, 2) / var)

def kl_div_var(q_mu, q_var, p_mu, p_var, eps=1e-12):

    return 0.5 * (torch.log((p_var+eps) / (q_var+eps)) + (q_var+eps) / (p_var+eps) + torch.pow(q_mu - p_mu, 2) / (p_var+eps) - 1)

def cosdis(x1,x2):
    return (1-torch.cosine_similarity(x1,x2,dim=-1))/2

def vade_trick(mc_sample, gmm_pi, gmm_mu, gmm_sca):

        log_pz_c = torch.mean(gaussian_log_var(mc_sample.unsqueeze(1), gmm_mu.unsqueeze(0), gmm_sca.unsqueeze(0)), dim=-1)

        log_pc = torch.log(gmm_pi.unsqueeze(0))
        log_pc_z = log_pc + log_pz_c
        pc_z = torch.exp(log_pc_z) + 1e-10
        normalize_pc_z = pc_z / torch.sum(pc_z, dim=1, keepdim=True)
        return normalize_pc_z

class Loss(nn.Module):
    def __init__(self, gamma=0.5, lamda=1):
        super(Loss, self).__init__()
        self.gamma = gamma
        self.lamda = lamda
    def compact_loss(self,u_list,var_list,mask):
        loss = 0
        view_num = len(u_list)
        for i in range(0,len(u_list)):
            for j in range(i+1,len(u_list)):

                # kl =  kl_div_var(u_list[i].detach(),var_list[i].detach(),u_list[j],var_list[j])
                # assert torch.sum(torch.isinf(kl)).item() == 0
                # kl_loss = kl * var_list[j]
                # kl_loss = kl * (mask[:,i]*mask[:,j]).unsqueeze(-1)

                kl = (u_list[i].detach()-u_list[j])**2
                kl = kl*var_list[j]
                kl_loss = kl*(mask[:,i]*mask[:,j]).unsqueeze(-1)
                if torch.sum(torch.isnan(kl_loss)).item() > 0:
                    pass
                # loss += kl_loss.sum(dim=-1).mean()
                loss += kl_loss.mean()
        assert torch.sum(torch.isnan(loss)).item() == 0
        loss = loss/view_num**2

        return loss
    def vae_loss_new(self,z_mu,z_sca, qc_x,  c_mu,c_sca,p_c,inc_L_ind):

        z_sam = gaussian_reparameterization_var(z_mu,z_sca)
        c_sam = gaussian_reparameterization_var(c_mu,c_sca)
        logsoft_qz_x = (z_sam.softmax(-1)+1e-5).log().detach()
        soft_qz_c = c_sam.softmax(-1)+1e-5
        # assert torch.sum(torch.isnan(kl_xc)).item() == 0


        kl_xc = torch.abs(F.kl_div(logsoft_qz_x.unsqueeze(1),soft_qz_c.unsqueeze(0),reduction='none'))
        if torch.sum(torch.isnan(kl_xc)).item() > 0:
            print('error')
        assert torch.sum(torch.isnan(kl_xc)).item() == 0

        z_kl_loss = torch.mean((qc_x.unsqueeze(-1) * kl_xc).sum((1,2)))


        # print((qc_x<0).sum())
        assert torch.sum(torch.isnan(z_kl_loss)).item() == 0
        if len(p_c.shape)==1:
            p_c = p_c.unsqueeze(0)
        c_kl_loss = torch.mean((qc_x * torch.log((qc_x / (p_c+1e-5)+1e-5))).abs())

        assert torch.sum(torch.isnan(c_kl_loss)).item() == 0
        # # print((c_kl_loss<0).sum())
        # c_kl_loss_mean = torch.mean(c_kl_loss)
        return z_kl_loss,c_kl_loss
    def z_c_loss(self,z_mu, label,  c_mu,inc_L_ind):
        loss = ((z_mu.unsqueeze(1)-c_mu.unsqueeze(0))**2).mean(dim=-1)

        loss = (loss*(label*inc_L_ind)).sum(-1)/((label*inc_L_ind).sum(-1)+1e-9)
        return loss.mean()
    def z_c_loss_new(self,z_mu, label,  c_mu,inc_L_ind):
        label_inc = label.mul(inc_L_ind)

        sample_label_emb = (label_inc.matmul(c_mu))/(label_inc.sum(-1)+1e-9).unsqueeze(-1)
        loss = ((z_mu-sample_label_emb)**2)
        # print(loss.mean())
        return loss.mean()
    def z_c_loss2(self,z_mu,z_sca, label,  c_mu,c_sca,inc_L_ind):
        # loss = (z_mu-(label*inc_L_ind).matmul(c_mu))**2
        loss = W_distance(z_mu.unsqueeze(1),z_sca.unsqueeze(1),c_mu.unsqueeze(0),c_sca.unsqueeze(0))
        loss = loss*(label*inc_L_ind)
        return loss.mean()
    def z_c_loss3(self,z_mu, label,  c_mu,inc_L_ind):
        # loss = (z_mu-(label*inc_L_ind).matmul(c_mu))**2
        loss = torch.abs(z_mu.unsqueeze(1)-c_mu.unsqueeze(0)).sum(-1)
        loss = (loss*(label*inc_L_ind)).sum(-1)/((label*inc_L_ind).sum(-1)+1e-9)
        return loss.sum()/z_mu.shape[0]

    def vae_loss(self,z_mu,z_sca, qc_x,  c_mu,c_sca,p_c,inc_L_ind):
        kl_xc = kl_div_var(z_mu.unsqueeze(1), z_sca.unsqueeze(1), c_mu.unsqueeze(0), c_sca.unsqueeze(0))
        # z_kl_loss = torch.mean(qc_x * torch.mean(kl, dim=-1),
        #                     dim=1)
        # z_kl_loss = torch.mean(z_kl_loss)

        # print(qc_x.log().sigmoid().unsqueeze(-1).shape,qz_c.log().sigmoid().unsqueeze(0).shape)
        # if torch.sum(torch.isnan(kl_xc)).item() > 0 or torch.sum(torch.isnan(kl_xc)).item()>0:
        #     assert torch.sum(torch.isnan(kl_xc)).item() == 0
        #     assert torch.sum(torch.isnan(kl_xc)).item() == 0

        # assert torch.sum(torch.isnan(kl_xc)).item() == 0

        if torch.sum(torch.isnan(kl_xc)).item() > 0:
            print('error')
        assert torch.sum(torch.isnan(kl_xc)).item() == 0

        z_kl_loss = torch.mean((qc_x.unsqueeze(-1) * kl_xc).sum((1,2)))


        # print((qc_x<0).sum())
        assert torch.sum(torch.isnan(z_kl_loss)).item() == 0
        if len(p_c.shape)==1:
            p_c = p_c.unsqueeze(0)
        c_kl_loss = torch.mean((qc_x * torch.log((qc_x / (p_c+1e-5)+1e-5))).abs())

        assert torch.sum(torch.isnan(c_kl_loss)).item() == 0
        # # print((c_kl_loss<0).sum())
        # c_kl_loss_mean = torch.mean(c_kl_loss)
        return z_kl_loss,c_kl_loss


    def tc_kl_loss(self,uniview_dist_mu, uniview_dist_sca, aggregate_mu, aggregate_sca, mask=None):
        z_tc_loss = []
        for v in range(len(uniview_dist_mu)):
            zv_tc_loss = torch.sum(kl_div_var(aggregate_mu, aggregate_sca, uniview_dist_mu[v], uniview_dist_sca[v]),
                                dim=1)
            exist_loss = zv_tc_loss * mask[v].squeeze()
            zv_tc_loss_mean = torch.sum(exist_loss)
            z_tc_loss.append(zv_tc_loss_mean)
        tc_term_loss = sum(z_tc_loss) / aggregate_mu.shape[0]

        return tc_term_loss

    def corherent_loss2(self,uniview_dist_mu, uniview_dist_sca, aggregate_mu, aggregate_sca, mask=None):
        if mask is None:
            mask = torch.ones_like((aggregate_mu.shape[0],len(uniview_dist_mu))).to(aggregate_mu.device)

        z_tc_loss = []
        # mask_stack = torch.concatenate(mask, dim=1)
        norm = torch.sum(mask, dim=1)
        for v in range(len(uniview_dist_mu)):
            zv_tc_loss = torch.mean((aggregate_mu - aggregate_sca)**2,dim=1)
            exist_loss = zv_tc_loss * mask[:,v]
            z_tc_loss.append(exist_loss)
        z_tc_loss = torch.stack(z_tc_loss,dim=1)

        sample_ave_tc_term_loss = torch.sum(z_tc_loss) / mask.sum()

        return sample_ave_tc_term_loss
    def corherent_loss(self,uniview_dist_mu, uniview_dist_sca, aggregate_mu, aggregate_sca, mask=None):
        if mask is None:
            mask = torch.ones_like((aggregate_mu.shape[0],len(uniview_dist_mu))).to(aggregate_mu.device)

        z_tc_loss = []
        # mask_stack = torch.concatenate(mask, dim=1)
        norm = torch.sum(mask, dim=1)
        weight = F.softmax(torch.stack(uniview_dist_sca,dim=1),dim=1)
        for v in range(len(uniview_dist_mu)):
            # zv_tc_loss = torch.mean(kl_div_var(aggregate_mu, aggregate_sca, uniview_dist_mu[v], uniview_dist_sca[v])*weight[:,v,:],
            #                     dim=1)
            zv_tc_loss = torch.mean(kl_div_var(aggregate_mu, aggregate_sca, uniview_dist_mu[v], uniview_dist_sca[v]),
                                dim=1)
            exist_loss = zv_tc_loss * mask[:,v]
            z_tc_loss.append(exist_loss)
        z_tc_loss = torch.stack(z_tc_loss,dim=1)

        sample_ave_tc_term_loss = torch.sum(z_tc_loss) / mask.sum()

        return sample_ave_tc_term_loss

    def weighted_BCE_loss(self,pred,label,inc_L_ind,reduction='mean'):
        assert torch.sum(torch.isnan(torch.log(pred))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - pred + 1e-5))).item() == 0
        res=torch.abs((label.mul(torch.log(pred + 1e-5)) \
                                                + (1-label).mul(torch.log(1 - pred + 1e-5))).mul(inc_L_ind))
        assert torch.sum(torch.isnan(res)).item() == 0
        assert torch.sum(torch.isinf(res)).item() == 0

        if reduction=='mean':
            return torch.sum(res)/torch.sum(inc_L_ind)
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res

    def weighted_wmse_loss(self,input, target, weight, reduction='mean'):
        ret = (torch.diag(weight).mm(target - input)) ** 2
        if torch.sum(torch.isnan(ret)).item()>0:
            print(ret)
        if reduction == 'mean':
            return torch.mean(ret)
        elif reduction=='sum':
            return torch.sum(ret)
        elif reduction=='none':
            return ret
    def weighted_cross_mse_loss(self, mask_v, data, rec_v):
        # data list [B,D]*V rec_v list[[V,B,D]*V]
        n_view = len(rec_v)
        mse1 = mse2 = 0
        for i in range(n_view):
            mse1 += torch.mean(torch.pow(data[i].detach() - rec_v[i][:, i, :], 2) * mask_v[:, i:i + 1])
            for j in range(n_view):
                if i != j:
                    mse2 += torch.mean(
                        torch.pow(data[i].detach() - rec_v[i][:, j, :], 2) * mask_v[:, i:i + 1] * mask_v[:, j:j + 1])
        return  mse1 + self.lamda * mse2 / n_view
    def contrast_loss(self, fea_emb, prot_emb, label,inc_L_ind):
        pos_label = label.mul(inc_L_ind)
        v1 = fea_emb
        v2 = prot_emb
        n = v1.size(0)
        pos_label_num = inc_L_ind.sum()

        if pos_label_num == 0:
            return 0

        v1 = F.normalize(v1, p=2, dim=1) #normalize two vectors
        v2 = F.normalize(v2, p=2, dim=1)
        # v2 = F.sigmoid(v2)
        sim_mat = torch.matmul(v1, v2.T) /self.gamma
        sim_mat_avai = sim_mat.masked_fill((1-inc_L_ind).bool(),-1e9)#M
        # denominator = torch.exp(sim_mat_avai).sum(dim=-1,keepdim=True)#M
        # # print(denominator.sum(-1))
        # assert torch.sum(torch.isnan(sim_mat_avai)).item() == 0
        # if (denominator==0).sum()>0:
        #     pass
        # sim_mat_avai_norm = torch.exp((sim_mat_avai)) / denominator
        sim_mat_avai_norm = torch.softmax((sim_mat_avai),dim=-1)
        assert torch.sum(torch.isnan(sim_mat_avai_norm)).item() == 0
        res = -((sim_mat_avai_norm+1e-9).log().mul(pos_label))
        # res = (res/(inc_L_ind.sum(dim=-1,keepdim=True)+1e-9)).sum()/n #M1
        res = res.sum()/pos_label.sum() #M2
        # res = res.sum()/n
        # print(res.max(),res.min())
        assert torch.sum(torch.isnan(res)).item() == 0
        return res
