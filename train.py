import numpy as np
import random
import torch
import os
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import argparse
import importlib
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import torch.backends.cudnn as cudnn
from torch.nn.functional import one_hot
from time import strftime
import re
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataloaders.datasets import ACDCDataset_Edge, MSCMRDataSets_Edge
from dataloaders.utils import *
from utils import pyutils
from utils.evaluate import test_single_volume_for_training
from utils import losses
from model.qemaxvit_unet import QEMaxViT_Unet

import gc
gc.collect()
torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--max_epoches", default=200, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--root_path', type=str,
                        default='/data/ACDC', help='Name of Experiment')
    parser.add_argument('--patch_size', type=list, default=[256, 256],
                        help='patch size of network input')
    parser.add_argument('--fold', type=str,
                        default='fold1', help='cross validation')
    parser.add_argument('--sup_type', type=str,
                        default='scribble', help='supervision type')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--exp', type=str,
                        default='ACDC/QEMaxViTUnet', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='QEMaxViTUnet', help='model_name')
    parser.add_argument('--num_classes', type=int,
                        default=4, help='output channel of network')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        help='learning rate schedule')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Specify the device to run the model on.')

    args = parser.parse_args()


    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    
    snapshot_path = "../model/{}_{}/{}".format(args.exp, args.fold, args.sup_type)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logdir = os.path.join(snapshot_path, "{}_log.txt".format(strftime("%Y_%m_%d_%H_%M_%S")))
    pyutils.Logger(logdir)
    print("log in ", logdir)
    print(vars(args))

    device = args.device

    model = QEMaxViT_Unet(num_classes=4, backbone_pretrained_pth="/teamspace/studios/this_studio/MIST/pretrained_pth/maxvit/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth").to(device)

    tblogger = SummaryWriter(args.tblog_dir)

    train_set = ACDCDataset_Edge(
        base_dir=args.root_path,
        split="train",
        transform=transforms.Compose([RandomGenerator([256,256], is_edge_mask=True)]),
        fold=args.fold,
        sup_type=args.sup_type,
        is_edge_mask=True,
    )

    val_set = ACDCDataset_Edge(
        base_dir=args.root_path,
        split='val',
        transform=None,
        fold=args.fold,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    batch_size = args.batch_size
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    num_classes = args.num_classes
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = losses.pDLoss(num_classes, ignore_index=4)
    edge_loss_function = nn.MSELoss()
    avg_meter = pyutils.AverageMeter('loss')
    best_performance = 0.0
    best_epoch = 0
    iter_num = 0
    max_iterations = args.max_epoches * len(trainloader)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    if args.lr_schedule == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100 * len(trainloader))

    for ep in range(args.max_epoches):
        for iter, sampled_batch in enumerate(trainloader):
            img, label, groundtruth_edge = sampled_batch['image'], sampled_batch['label'], sampled_batch['edge_mask']
            img, label, groundtruth_edge = img.to(device), label.to(device), groundtruth_edge.to(device)
            output_main, output_aux, edge_map = model(img)
            outputs_soft1 = torch.softmax(output_main, dim=1)
            outputs_soft2 = torch.softmax(output_aux, dim=1)
            beta = random.random() + 1e-10
            pseudo_supervision = torch.argmax(
                (beta * outputs_soft1.detach() + (1.0-beta) * outputs_soft2.detach()), dim=1, keepdim=False)
            loss_pse_sup = 0.5 * (dice_loss(outputs_soft1, pseudo_supervision.unsqueeze(
                1)) + dice_loss(outputs_soft2, pseudo_supervision.unsqueeze(1)))

            loss_ce1 = ce_loss(output_main, label[:].long())
            loss_ce2 = ce_loss(output_aux, label[:].long())
            loss_ce = 0.5 * (loss_ce1 + loss_ce2)

            edge_loss = edge_loss_function(edge_map, groundtruth_edge)
            loss = loss_ce + 0.5 * loss_pse_sup + 0.2 * edge_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_meter.add({'loss': loss.item()})

            scheduler.step()
            iter_num += 1

        else:
            print('epoch: %5d' % ep, 'loss: %.4f' % avg_meter.get('loss'), flush=True)
            model.eval()
            metric_list = []
            for i_batch, sampled_batch in enumerate(valloader):
                metric_i = test_single_volume_for_training(
                sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                metric_list.append(metric_i)
            metric_list = np.nanmean(np.array(metric_list), axis=0)
            for class_i in range(num_classes - 1):
                tblogger.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                  metric_list[class_i, 0], ep)
                tblogger.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                  metric_list[class_i, 1], ep)


            performance = np.mean(metric_list, axis=0)[0]
            mean_hd95 = np.mean(metric_list, axis=0)[1]

            tblogger.add_scalar('info/val_mean_dice', performance, ep)
            tblogger.add_scalar('info/val_mean_hd95', mean_hd95, ep)

            if performance > 0.85:
                print("Update high dice score model!")
                file_name = os.path.join(snapshot_path, '{}_{}_model.pth'.format(args.model, str(performance)[0:6]))
                torch.save(model.state_dict(), file_name)

            if (ep+1) % 100 == 0:
                print("{} model!".format(ep))
                file_name = os.path.join(snapshot_path, '{}_{}_model.pth'.format(args.model, ep))
                torch.save(model.state_dict(), file_name)

            if performance > best_performance:
                best_performance = performance
                best_epoch = ep
                save_best = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                torch.save(model.state_dict(), save_best) 
                print('best model in epoch %5d  mean_dice : %.4f' % (ep, performance))
            
            print('epoch %5d  mean_dice : %.4f mean_hd95 : %.4f' % (ep, performance, mean_hd95), flush=True)

            model.train()
            avg_meter.pop()

    print('best model in epoch %5d  mean_dice : %.4f' % (best_epoch, best_performance))
    print('save best model in {}/{}_best_model.pth'.format(snapshot_path, args.model))
    torch.save(model.state_dict(), os.path.join(snapshot_path,'{}_final_model.pth'.format(args.model)))