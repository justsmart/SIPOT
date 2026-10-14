import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import utils
from utils import AverageMeter
import MLdataset
import argparse
import time
from model_imput import get_model
import evaluation
import torch
import numpy as np
import copy
from myloss import Loss
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR



def view_mixup(data_list,label,v_mask,l_mask):
    new_data = []
    new_labels = label.unsqueeze(0).repeat(len(data_list),1,1)
    num = data_list[0].shape[0]
    for i,view_data in enumerate(data_list):
        indices = torch.randperm(num).cuda()
        new_data.append(torch.index_select(view_data, 0, indices))
        new_labels[i] = torch.index_select(new_labels[i],0,indices)
        v_mask[:,i] = torch.index_select(v_mask[:,i],0,indices)

    label = new_labels.sum(dim=0)
    label = torch.masked_fill(label,label>0,1)
    l_mask = torch.masked_fill(l_mask,label>0,1)
    return new_data,label,v_mask,l_mask

def train(loader, model, loss_model, opt, sche, epoch,dep_graph,last_preds,logger):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    mce= nn.MultiLabelSoftMarginLoss()
    model.train()
    end = time.time()
    All_preds = torch.tensor([]).cuda()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)
        data=[v_data.to('cuda:0') for v_data in data]
        label = label.to('cuda:0')
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')

        if last_preds is None:
            last_p = label*inc_L_ind
        else:
            last_p = last_preds[i*args.batch_size:i*args.batch_size+label.shape[0]].detach()
        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list_cross, label_emb_sample, pred, label_emb,label_emb_var,new_p = model(data,mask=inc_V_ind )
        All_preds = torch.cat([All_preds,new_p],dim=0)












        loss_CL = loss_model.weighted_BCE_loss(pred,label,inc_L_ind) + loss_model.weighted_BCE_loss(new_p,label,inc_L_ind)
        contrast_loss = loss_model.contrast_loss(z_sample, label_emb_sample, label, inc_L_ind)
        cohr_loss = loss_model.corherent_loss(uniview_mu_list, uniview_sca_list,fusion_z_mu, fusion_z_sca,mask=inc_V_ind)
        loss_mse = loss_model.weighted_cross_mse_loss(inc_V_ind,data,xr_list_cross)
        assert torch.sum(torch.isnan(loss_mse)).item() == 0
        loss = loss_CL + loss_mse *args.alpha + contrast_loss*args.beta + cohr_loss *args.sigma
        opt.zero_grad()
        loss.backward()
        if isinstance(sche,CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))

        opt.step()
        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    if isinstance(sche,StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, losses=losses))
    return losses,model,All_preds,label_emb_sample

def test(loader, model, loss_model, epoch,logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    model.eval()
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data=[v_data.to('cuda:0') for v_data in data]

        z_sample, uniview_mu_list, uniview_sca_list, fusion_z_mu, fusion_z_sca, xr_list, label_embedding_sample, pred, label_emb,label_emb_var, new_p = model(data,mask=inc_V_ind.to('cuda:0'))

        pred = pred.cpu()
        total_labels = np.concatenate((total_labels,label.numpy()),axis=0) if len(total_labels)>0 else label.numpy()
        total_preds = np.concatenate((total_preds,pred.detach().numpy()),axis=0) if len(total_preds)>0 else pred.detach().numpy()

        loss = loss_model.weighted_BCE_loss(pred,label,inc_L_ind)

        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    total_labels=np.array(total_labels)
    total_preds=np.array(total_preds)

    evaluation_results=evaluation.do_metric(total_preds,total_labels)
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}\t'
                  'AP {ap:.3f}\t'
                  'HL {hl:.3f}\t'
                  'RL {rl:.3f}\t'
                  'AUC {auc:.3f}\t'.format(
                        epoch,   batch_time=batch_time,
                        losses = losses,
                        ap=evaluation_results[0],
                        hl=evaluation_results[1],
                        rl=evaluation_results[2],
                        auc=evaluation_results[3]
                        ))
    return evaluation_results

def main(args,file_path):
    data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view.mat')
    fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view_MaskRatios_' + str(
                                args.mask_view_ratio) + '_LabelMaskRatio_' +
                                str(args.mask_label_ratio) + '_TraindataRatio_' +
                                str(args.training_sample_ratio) + '.mat')

    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(9)]
    logger = utils.setLogger(None)
    device = torch.device('cuda:0')
    for fold_idx in range(folds_num):
        fold_idx=fold_idx
        train_dataloder,train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='train',batch_size=args.batch_size,shuffle = True,num_workers=4)
        test_dataloder,test_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,val_ratio=0.15,fold_idx=fold_idx,mode='test',batch_size=args.batch_size,num_workers=4)
        val_dataloder,val_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='val',batch_size=args.batch_size,num_workers=4)
        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num
        labels = torch.tensor(train_dataset.cur_labels).float().to('cuda:0')
        dep_graph = None
        pri_c = train_dataset.cur_labels.sum(axis=0)/train_dataset.cur_labels.shape[0]
        pri_c = torch.tensor(pri_c).cuda()
        model=get_model(d_list,num_classes=classes_num,z_dim=args.z_dim,adj=dep_graph,rand_seed=0)
        loss_model = Loss(args.gamma,args.lamda)
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = None


        logger.info('train_data_num:'+str(len(train_dataset))+'  test_data_num:'+str(len(test_dataset))+'   fold_idx:'+str(fold_idx))
        print(args)
        static_res = 0
        epoch_results = [AverageMeter() for i in range(9)]
        total_losses = AverageMeter()
        train_losses_last = AverageMeter()
        best_epoch=0
        best_model_dict = {'model':model.state_dict(),'epoch':0}
        for epoch in range(args.epochs):
            if epoch==0:
                All_preds = None
            train_losses,model,All_preds,label_emb_sample = train(train_dataloder,model,loss_model,optimizer,scheduler,epoch,dep_graph,All_preds,logger)

            if epoch>=args.pre_epochs:
                val_results = test(val_dataloder,model,loss_model,epoch,logger)

                if val_results[0]*0.25+val_results[2]*0.25+val_results[3]*0.25>=static_res:
                    static_res = val_results[0]*0.25+val_results[2]*0.25+val_results[3]*0.25
                    best_model_dict['model'] = copy.deepcopy(model.state_dict())
                    best_model_dict['epoch'] = epoch
                    best_epoch=epoch
                train_losses_last = train_losses
                total_losses.update(train_losses.sum)
        model.load_state_dict(best_model_dict['model'])
        print("epoch",best_model_dict['epoch'])
        test_results = test(test_dataloder,model,loss_model,epoch,logger)
        logger.info('final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(fold_idx,best_epoch,test_results[0],test_results[1],
            test_results[2],test_results[3]))

        for i in range(9):
            folds_results[i].update(test_results[i])
    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write(
            'AP 1-HL 1-RL AUCme 1-oneE 1-Cov macAUC macro_f1 micro_f1 lr alpha beta gamma sigma lamda\n')
    res_list = [str(round(res.avg,3))+'+'+str(round(res.std,3)) for res in folds_results]
    res_list.extend([str(args.lr),str(args.alpha),str(args.beta),str(args.gamma),str(args.sigma),str(args.lamda)])
    res_str = ' '.join(res_list)
    file_handle.write(res_str)
    file_handle.write('\n')
    file_handle.close()


def filterparam(file_path,index):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines)>1 else []
        params = [[float(line.split(' ')[idx]) for idx in index] for line in lines ]
    return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--records-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'final_records'))
    parser.add_argument('--file-path', type=str, metavar='PATH',
                        default='')
    parser.add_argument('--root-dir', type=str, metavar='PATH',
                        default='./data/')
    parser.add_argument('--dataset', type=str, default='')#mirflickr corel5k pascal07 iaprtc12 espgame
    parser.add_argument('--datasets', type=list, default=['corel5k'])
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=10, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)

    parser.add_argument('--name', type=str, default='final_Cross_')
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=150) #150 for corel5k; 100 for iaprtc12,espgame; 20 for pascal07,mirflickr
    parser.add_argument('--pre_epochs', type=int, default=0)
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--beta', type=float, default=1e-1)
    parser.add_argument('--gamma', type=float, default=1e-1)
    parser.add_argument('--sigma', type=float, default=0.)
    parser.add_argument('--lamda', type=float, default=0.)

    args = parser.parse_args()

    os.makedirs(args.records_dir, exist_ok=True)
    lr_list = [1e-3]
    alpha_list = [1e0]#
    beta_list = [1e-3]# 1e-3 for corel5k 1e-2 for others; for no missing, 1e-1 for PAS and MIR dataset, 1e-2 for COR and ESP, 1e0 for IAP
    gamma_list = [0.5]# 0.5 for all except for MIR with 'no missing' (0 is recommand)
    sigma_list = [1e0]# 1 for all except for IAP/COR with 'no missing' (0.1 is recommand)
    lamda_list = [1e0]# 1 for 0.5 missing rate; 0.1 for no missing rate.
    if args.lr >= 0.01:
        args.momentumkl = 0.90
    for lr in lr_list:
        args.lr = lr
        for alpha in alpha_list:
            args.alpha = alpha
            for beta in beta_list:
                args.beta = beta
                for gamma in gamma_list:
                    args.gamma = gamma
                    for sigma in sigma_list:
                        args.sigma = sigma
                        for lamda in lamda_list:
                            args.lamda = lamda
                            for dataset in args.datasets:
                                args.dataset = dataset
                                file_path = osp.join(args.records_dir,args.name+args.dataset+'_VM_' + str(
                                                args.mask_view_ratio) + '_LM_' +
                                                str(args.mask_label_ratio) + '_T_' +
                                                str(args.training_sample_ratio) + '.txt')
                                args.file_path = file_path
                                existed_params = filterparam(file_path,[-5,-4,-3,-2,-1])
                                if [args.lr,args.alpha,args.beta,args.gamma,args.sigma] in existed_params:
                                    print('existed param! beta:{}'.format(args.beta))
                                main(args,file_path)
