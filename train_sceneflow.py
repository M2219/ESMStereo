from __future__ import print_function, division

import os
import time
import gc
import sys

import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from datasets import __datasets__
from models import __models__, model_loss_train, model_loss_test
from utils import *

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='ESMStereo')
parser.add_argument('--model', default='ESMStereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--backbone', default='efficientnet_b2', help='select a model structure', choices=["mobilenetv2_100", "efficientnet_b2"])
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/datasets/SceneFlow/flyingthings3d", help='data path')
parser.add_argument('--trainlist', default='./filenames/sceneflow_train.txt', help='training list')
parser.add_argument('--testlist',default='./filenames/sceneflow_test.txt', help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
#parser.add_argument('--lr', type=float, default=0.0002, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
#parser.add_argument('--epochs', type=int, default=80, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default="20,32,40,48,56:2", help='the epochs to decay lr: the downscale rate')
#parser.add_argument('--lrepochs', type=str, default="10,20,30,40,50,60,70:2", help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default='', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default='', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--performance', action='store_true', help='evaluate the performance')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=1, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--cv_scale', type=int, default=4, help='cost volume scale factor', choices=[16, 8, 4])
parser.add_argument('--cv', type=str, default='gwc', choices=[
          'norm_correlation',
          'gwc',
], help='selecting a cost volumes')

args = parser.parse_args()

gwc = False
norm_correlation = False
if args.cv == 'norm_correlation':
    norm_correlation = True
elif args.cv == 'gwc':
    gwc = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

print("creating new summary file")
logger = SummaryWriter(args.logdir)

StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=8, drop_last=False)

model = __models__[args.model](args.maxdisp, gwc, norm_correlation, args.backbone, args.cv_scale)
model = nn.DataParallel(model)
model.cuda()



print("The number of parameters:", count_parameters(model))
optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

start_epoch = 0
if args.resume:
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, weights_only=False)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)

print("start at epoch {}".format(start_epoch))

def train():
    bestepoch = 0
    error = 100

    loss_ave = AverageMeter()
    EPE_ave = AverageMeter()
    D1_ave = AverageMeter()
    loss_ave_t = AverageMeter()
    EPE_ave_t = AverageMeter()
    D1_ave_t = AverageMeter()

    if args.performance:
        dummy_input1 = torch.randn(1, 3, 512, 960, dtype=torch.float).cuda()
        dummy_input2 = torch.randn(1, 3, 512, 960, dtype=torch.float).cuda()
        inference_time = measure_performance(dummy_input1, dummy_input2)
        print("inference time = ", inference_time)
        return 0

    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs = train_sample(sample, compute_metrics=do_summary)

            loss_ave.update(loss)
            EPE_ave.update(scalar_outputs['EPE'][0])
            D1_ave.update(scalar_outputs['D1'][0])

            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)

            print('Epoch {}/{} | Iter {}/{} | train loss = {:.3f}({:.3f}) | EPE = {:.3f}({:.3f}) | D1 = {:.3f}({:.3f}) | time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss, loss_ave.avg,
                                                                                       scalar_outputs['EPE'][0], EPE_ave.avg, scalar_outputs['D1'][0], D1_ave.avg,
                                                                                       time.time() - start_time))
            del scalar_outputs
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            start_time = time.time()
            loss, scalar_outputs = test_sample(sample, compute_metrics=do_summary)
            tt = time.time()

            loss_ave_t.update(loss)
            EPE_ave_t.update(scalar_outputs['EPE'][0])
            D1_ave_t.update(scalar_outputs['D1'][0])

            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)

            print('Epoch {}/{} | Iter {}/{} | test loss = {:.3f}({:.3f}) | EPE = {:.3f}({:.3f}) | D1 = {:.3f}({:.3f}) | time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TestImgLoader), loss, loss_ave_t.avg,
                                                                                       scalar_outputs['EPE'][0], EPE_ave_t.avg, scalar_outputs['D1'][0], D1_ave_t.avg,
                                                                                       tt - start_time))
            del scalar_outputs

        avg_test_scalars = avg_test_scalars.mean()
        nowerror = avg_test_scalars["EPE"][0]
        if  nowerror < error :
            bestepoch = epoch_idx
            error = avg_test_scalars["EPE"][0]
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
        gc.collect()
    print('MAX epoch %d total test error = %.5f' % (bestepoch, error))

def train_sample(sample, compute_metrics=False):
    model.train()
    imgL, imgR, disp_gt, disp_gt_low = sample['left'], sample['right'], sample['disparity'], sample['disparity_low']

    imgL = imgL.cuda()
    imgR = imgR.cuda()

    disp_gt = disp_gt.cuda()
    disp_gt_low = [d.cuda() for d in disp_gt_low]

    optimizer.zero_grad()

    disp_ests = model(imgL, imgR, train_status=True)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)

    mask_low = [(d < args.maxdisp) & (d > 0) for d in disp_gt_low]
    masks = [mask] + mask_low
    disp_gts = [disp_gt] + disp_gt_low

    loss = model_loss_train(disp_ests, disp_gts, masks, args.cv_scale)
    disp_ests_final = [disp_ests[0]]

    scalar_outputs = {"loss": loss}
    if compute_metrics:
        with torch.no_grad():
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests_final]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests_final]

    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs)

@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt= sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    disp_ests = model(imgL, imgR, train_status=False)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    masks = [mask]
    disp_gts = [disp_gt]
    loss = model_loss_test(disp_ests, disp_gts, masks)

    scalar_outputs = {"loss": loss}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs)

@make_nograd_func
def measure_performance(dummy_input1, dummy_input2):
    model.eval()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 500
    timings=np.zeros((repetitions,1))
    for _ in range(10):
        _ = model(dummy_input1, dummy_input2, train_status=True)

    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input1, dummy_input2, train_status=True)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    np.std(timings)

    return  mean_syn

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    train()
