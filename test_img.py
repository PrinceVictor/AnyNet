import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from utils import preprocess
import utils.logger as logger
import torch.backends.cudnn as cudnn
import argparse
import models.anynet
from PIL import Image
import numpy as np
import time
import os
import glob
import sys

parser = argparse.ArgumentParser(description='inference img')
parser.add_argument('--loadmodel', default='/home/victor/mobile_robot/AnyNet/para/checkpoint/kitti2015_ck/checkpoint.tar',
                    help='loading model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--save_path', type=str, default='results/finetune_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--with_cspn', action='store_true', help='with cspn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                    help='pretrained model path')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda == True:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print('args spn {}'.format(args.with_spn))

model = models.anynet.AnyNet(args)
model = nn.DataParallel(model).cuda()

# if args.loadmodel is not None:
#     print('load AnyNet')
#     state_dict = torch.load(args.loadmodel)
#     model.load_state_dict(state_dict['state_dict'], strict=False)

if __name__ == '__main__':

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    print(model)
    # sys.exit()

    process = preprocess.get_transform(augment=False)

    left_images_dir = '/home/victor/dataset/data_scene_flow/testing/image_2'
    right_images_dir = '/home/victor/dataset/data_scene_flow/testing/image_3'
    if os.path.isdir(left_images_dir):
        left_image_paths = glob.glob(os.path.join(left_images_dir, '*.png'))
        print('left images num: ', len(left_image_paths))
        right_image_paths = glob.glob(os.path.join(right_images_dir, '*.png'))
        print('right images num: ', len(left_image_paths))
        # if len(image_paths)==0:
        #     print("non image input!!!!")
        #     break
    else:
        left_image_paths = [left_images_dir]
        right_image_paths = [right_images_dir]

    model.eval()
    # print(model)

    for i in range(len(left_image_paths)):

        imgL = Image.open(left_image_paths[i])
        imgR = Image.open(right_image_paths[i])

        w, h = imgL.size

        imgL = imgL.crop((w - 1232, h - 368, w, h))
        imgR = imgR.crop((w - 1232, h - 368, w, h))

        imgL = process(imgL).unsqueeze(0)
        imgR = process(imgR).unsqueeze(0)
        imgL = imgL.contiguous()
        imgR = imgR.contiguous()

        start_time = time.time()
        with torch.no_grad():
            outputs = model(imgL, imgR)
        inference_time = time.time() - start_time
        print('i:{} inference time {:.3f}ms FPS {}'.format(i, inference_time*1000, round(1/inference_time)))
        # print(len(outputs))

        # print(len(outputs))
        # output = outputs[3].squeeze().cpu()

        # break
        # disparity = output.numpy()
        # disparity = (disparity).astype('uint8')
        # disparity = Image.fromarray(disparity)
        # disparity.save('results/' + left_image_paths[i][-13:])



