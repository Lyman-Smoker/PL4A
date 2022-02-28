import numpy as np
import os
from tqdm import tqdm
from scipy import stats
import random
import argparse
import time
import copy
# dataset
from seven_dataset import Seven_Dataset
# model
from models.I3D_Backbone import I3D_backbone
from models.Pattern_Miner import build_pattern_miner
from models.Classification_Branch import Cls_Branch
from models.Regression_Branch import Rgs_Branch
from models.PL4A import PL4A
# optimization
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
# visualization
from torch.utils.tensorboard import SummaryWriter
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
# helpers and args
from opts import *
from helpers import fix_bn, get_video_trans
from config import get_parser


# set cuda seeds
torch.manual_seed(0);
torch.cuda.manual_seed_all(0);
random.seed(0);
np.random.seed(0)
torch.backends.cudnn.deterministic = True

def buid_dataloaders():
    train_trans, test_trans = get_video_trans()
    train_dataset = Seven_Dataset(subset='train', multi_action=True, transform=train_trans)
    test_dataset = Seven_Dataset(subset='test', multi_action=True, transform=test_trans)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                            shuffle=True,num_workers = args.num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                            shuffle=False,num_workers = args.num_workers)
    return train_dataloader, test_dataloader

def buid_models():
    i3d = I3D_backbone(I3D_class=400, gap=False)
    pl4a = PL4A(pattern_dim=1024, num_pattern=20, cls_out_dim=len(action_id_list), tau=1, topK=args.topk, drop_rate=args.drop_rate)
    return i3d, pl4a


def run_net():
    print('Start running ...')

    # models
    i3d, pl4a = buid_models()
    patterns = patterns.float()

    # load pretrained models or checkpoint if necessary
    i3d.load_pretrain(args.pretrained_i3d_weight)

    # optimizer & scheduler
    optimizer = optim.Adam([
        {'params': i3d.parameters(), 'lr': args.base_lr * args.lr_factor},
        {'params': pl4a}
    ], lr=args.base_lr, weight_decay=args.weight_decay)
    scheduler = None

    # parameter setting
    start_epoch = 0
    global epoch_best, rho_best, L2_min, RL2_min
    epoch_best = 0
    rho_best = 0
    acc_best = 0.0
    L2_min = 1000
    RL2_min = 1000

    # loss
    mse = nn.MSELoss().cuda()
    ce = nn.CrossEntropyLoss().cuda()

    # cuda & DP
    patterns = patterns.cuda()
    i3d = i3d.cuda()
    cls_branch = cls_branch.cuda()
    rgs_branch = rgs_branch.cuda()

    i3d = nn.DataParallel(i3d)
    cls_branch = nn.DataParallel(cls_branch)
    rgs_branch = nn.DataParallel(rgs_branch)

    # dataset & loader
    train_dataloader, test_dataloader = buid_dataloaders()

    # train & test
    steps = ['train', 'test']
    for epoch in range(start_epoch, args.num_epochs):
        print('EPOCH:', epoch)
        
        for step in steps:
            print(step + ' step:')
            with open('./log.txt', 'a') as f:
                f.write('epoch %d - %s' % (epoch, step))
                f.write('\n')
            true_scores = [[] for _ in range(len(action_id_list))]
            pred_scores = [[] for _ in range(len(action_id_list))]
            true_class_count = [0 for _ in range(len(action_id_list))]
            pred_class_count = [0 for _ in range(len(action_id_list))]
            if args.fix_bn:
                i3d.apply(fix_bn)  # fix bn
            if step == 'train':
                i3d.train()
                cls_branch.train()
                rgs_branch.train()
                torch.set_grad_enabled(True)
                data_loader = train_dataloader
            else:
                i3d.eval()
                cls_branch.eval()
                rgs_branch.eval()
                torch.set_grad_enabled(False)
                data_loader = test_dataloader
            for data in tqdm(data_loader):
                video = data['video'].float().cuda()
                final_score = data['final_score'].float().reshape(-1, 1).cuda()
                action_idx = data['action_idx'].cuda()
                batch_size = video.shape[0]

                # Forward
                # 1: pass i3d backbone
                i3d_feats = i3d(video)  # B, 10, 1024
                # 2: pass pl4a
                cls_result, pred_score = pl4a(i3d_feats)
                

                # Loss
                loss = 0.0
                loss += args.alpha * mse(pred_score, final_score)
                loss += args.beta * ce(cls_result, action_idx)
                # print(loss)
                # Optimization
                if step == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # Updating score lists
                for b in range(batch_size):
                    action = action_idx[b]
                    # print(action)
                    true_class_count[action] += 1
                    if action == torch.argmax(cls_result[b]):
                        pred_class_count[action] += 1
                    true_scores[action].append(data['final_score'][b].numpy())
                    pred_scores[action].append(pred_score[b].item())
                    

            # analysis on results
            classes_name = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
            global_pred_scores = []
            global_true_scores = []
            # 1. each action
            print(step + ' results:')
            for i in range(len(action_id_list)):
                global_pred_scores.extend(pred_scores[i])
                global_true_scores.extend(true_scores[i])
                pred_s = np.array(pred_scores[i])
                true_s = np.array(true_scores[i])
                rho, p = stats.spearmanr(pred_s, true_s)
                L2 = np.power(pred_s - true_s, 2).sum() / true_s.shape[0]
                RL2 = np.power((pred_s - true_s) / (true_s.max() - true_s.min()), 2).sum() / \
                    true_s.shape[0]
                acc = pred_class_count[i] / true_class_count[i]
                print('\t%15s:\t\t\t acc-%.4f, cor-%.4f, L2-%.4f, RL2-%.4f' % (classes_name[i], acc, rho, L2, RL2))
                with open('./log.txt', 'a') as f:
                    f.write('\t%15s:\t\t\t acc-%.4f, cor-%.4f, L2-%.4f, RL2-%.4f\n' % (classes_name[i], acc, rho, L2, RL2))
            # 2. global
            global_acc = sum(pred_class_count) / sum(true_class_count)
            global_pred_scores = np.array(global_pred_scores)
            global_true_scores = np.array(global_true_scores)
            global_rho, global_p = stats.spearmanr(global_pred_scores, global_true_scores)
            global_L2 = np.power(global_pred_scores - global_true_scores, 2).sum() / global_true_scores.shape[0]
            global_RL2 = np.power((global_pred_scores - global_true_scores) / (global_true_scores.max() - global_true_scores.min()), 2).sum() / \
                global_true_scores.shape[0]
            print('--------------------------------------------------------------------------------')
            print('\tglobal:\t\t\t acc-%.4f, cor-%.4f, L2-%.4f, RL2-%.4f' % (global_acc, global_rho, global_L2, global_RL2))
            with open('./log.txt', 'a') as f:
                f.write('\tglobal:\t\t\t acc-%.4f, cor-%.4f, L2-%.4f, RL2-%.4f\n' % (global_acc, global_rho, global_L2, global_RL2))
                f.write('\n')
            
            if step == 'test':
                print('Current best:\t\t\t acc-%.4f, cor-%.4f, RL2-%.4f at epoch %d' % (acc_best, rho_best, RL2_min, epoch_best))
                if global_rho > rho_best:
                    print('___________________find new best___________________')
                    rho_best = global_rho
                    acc_best = global_acc
                    RL2_min = global_RL2
                    epoch_best = epoch
                    # save_checkpoint(base_model, bidir_attention, ln_mlp, regressor, optimizer, epoch, rho, RL2,
                    #                 args.exp_name + '_%.4f_%.4f@_%d' % (rho, RL2, epoch))
            print('--------------------------------------------------------------------------------')
    return




if __name__ == '__main__':
    args = get_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open('./log.txt', 'w') as f:
        f.write('\n')
    # set where to save exp results
    run_net()
    