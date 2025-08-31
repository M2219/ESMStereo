import cv2
import os
import argparse
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from models import __models__

torch.backends.cudnn.benchmark = True
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ESMStereo_confidence')
    parser.add_argument('--model', default='ESMStereo_confidence', help='select a model structure', choices=__models__.keys())
    parser.add_argument('--backbone', default='mobilenetv2_100', help='select a model structure', choices=["mobilenetv2_100", "efficientnet_b2"])
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
    parser.add_argument('--datapath', default="/datasets/kitti_2015/training/", help='data path')
    parser.add_argument('--kitti', type=str, default='2015')
    parser.add_argument('--loadckpt', default='./checkpoint/esmstereo_S_gwc_confidence.ckpt', help='load the weights from a specific checkpoint')
    parser.add_argument('--cv_scale', type=int, default=16, help='cost volume scale factor', choices=[16, 8, 4])
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

    MAXDISP=192
    CHECKPOINT=args.loadckpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = __models__[args.model](args.maxdisp, gwc, norm_correlation, args.backbone, args.cv_scale, device)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    state_dict = torch.load(CHECKPOINT)
    model.load_state_dict(state_dict["model"])
    print("number of parameters == ", count_parameters(model))

    left = torch.ones((1, 3, 384, 1248)).cuda()
    right = torch.ones((1, 3, 384, 1248)).cuda()

    onnx = torch.onnx.export(model.module, args=(left, right), f="StereoModelConf.onnx", input_names=["left", "right"], output_names=["disp", "confidence"], verbose=True, do_constant_folding=True)
