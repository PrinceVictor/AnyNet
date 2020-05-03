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
import cv2 as cv
import socket
import image_msg_pb2 as msg
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

# if args.cuda == True:
#     device = torch.device("cuda:0")
# else:
#     device = torch.device("cpu")

# print('args spn {}'.format(args.with_spn))

model = models.anynet.AnyNet(args)
model = nn.DataParallel(model).cuda()
model = model.cuda()

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

if args.loadmodel is not None:
    print('load AnyNet')
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'], strict=False)

if __name__ == '__main__':

    process = preprocess.get_transform(augment=False)

    # left_images_dir = '/home/victor/dataset/data_scene_flow/testing/image_2'
    # right_images_dir = '/home/victor/dataset/data_scene_flow/testing/image_3'

    left_images_dir = '/home/victor/dataset/2011_09_26/2011_09_26_drive_0005_sync/image_02/data'
    right_images_dir = '/home/victor/dataset/2011_09_26/2011_09_26_drive_0005_sync/image_03/data'

    # left_images_dir = '/home/liupengchao/zhb/dataset/Kitti/data_scene_flow/testing/image_2'
    # right_images_dir = '/home/liupengchao/zhb/dataset/Kitti/data_scene_flow/testing/image_3'

    if os.path.isdir(left_images_dir):
        left_image_paths = sorted(glob.glob(os.path.join(left_images_dir, '*.png')))
        print('left images num: ', len(left_image_paths))
        right_image_paths = sorted(glob.glob(os.path.join(right_images_dir, '*.png')))
        print('right images num: ', len(left_image_paths))
        # if len(image_paths)==0:
        #     print("non image input!!!!")
        #     break
    else:
        left_image_paths = [left_images_dir]
        right_image_paths = [right_images_dir]

    model.eval()

    # image_msg = msg.image()
    image_msg_buf = msg.image_buf()
    disp_buf = image_msg_buf.image_.add()
    color_buf = image_msg_buf.image_.add()
    # for name, param in model.named_parameters():
    #     print('name {}, para {}\n'.format(name, param))
    ip_port = ('127.0.0.1', 1080)
    tcp_socket = socket.socket()
    tcp_socket.connect(ip_port)

    start = "start1"
    flag = tcp_socket.recv(6)
    steps =1

    for i in range(len(left_image_paths)):
        print("start time {}" .format(time.time()))
        # print(left_image_paths[i])
        # print(left_image_paths[i])

        imgL = Image.open(left_image_paths[i])
        imgR = Image.open(right_image_paths[i])

        w, h = imgL.size

        imgL = imgL.crop((w - 1232, h - 368, w, h))
        imgR = imgR.crop((w - 1232, h - 368, w, h))

        imgL_tensor = process(imgL).unsqueeze(0)
        imgR_tensor = process(imgR).unsqueeze(0)

        imgL_tensor = imgL_tensor.contiguous().cuda()
        imgR_tensor = imgR_tensor.contiguous().cuda()

        start_time = time.time()
        with torch.no_grad():
            outputs = model(imgL_tensor, imgR_tensor)
        inference_time = time.time() - start_time
        print('i:{} inference time {:.3f}ms FPS {}'.format(i, inference_time*1000, round(1/inference_time)))
        # break

        # print(len(outputs))
        output = outputs[3].squeeze().cpu()

        disparity = output.numpy()
        disparity = (disparity).astype('uint8')

        left_color = np.asarray(imgL)
        left_color = left_color[:, :, ::-1]

        print("steps {}" .format(steps))
        steps = steps+1
        # print("start start time {}".format(time.time()))
        length = tcp_socket.sendall(start.encode())
        # print("start finish time {}".format(time.time()))
        # print(len(start.encode()))

        h, w = disparity.shape
        disp_buf.width = w
        disp_buf.height = h
        disp_buf.channel = 1
        disp_buf.size = w * h * 1
        disp_buf.mat_data = disparity.tostring()
        disp_buf.time_stamp = time.time()

        h, w, c = left_color.shape
        color_buf.width = w
        color_buf.height = h
        color_buf.channel = c
        color_buf.size = w * h * c
        color_buf.mat_data = left_color.tostring()
        color_buf.time_stamp = time.time()

        # print(color_buf.time_stamp)
        # print("image start time {}".format(time.time()))
        stream = image_msg_buf.SerializeToString()
        length = tcp_socket.send(stream)
        # print(length)
        print('end time {}' .format(time.time()))
        print()
        # print(len(stream))

        # left_color = imgL.squeeze().cpu().numpy()
        # left_color = cv.imread(left_image_paths[i])
        # left_color = left_color.reshape(left_color.shape[1], left_color.shape[2], -1)
        # left_color = left_color[:, :, ::-1].astype('uint8')
        # cv.imshow('left', left_color)
        # depth_image = cv.applyColorMap(cv.convertScaleAbs(disparity, alpha=1, beta=0),
        #                                 cv.COLORMAP_JET)
        depth_image = cv.applyColorMap(disparity,
                                       cv.COLORMAP_JET)
        #
        combined = np.zeros((left_color.shape[0]*2, left_color.shape[1], left_color.shape[2]), dtype=np.uint8)
        combined[:left_color.shape[0], :,:] = left_color
        combined[left_color.shape[0]:, :, :] = depth_image
        cv.imshow('combine', combined)
        cv.waitKey(10)
        # disparity = Image.fromarray(disparity)
        # disparity.save('results/' + left_image_paths[i][-13:])



