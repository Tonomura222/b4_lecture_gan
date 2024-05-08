# 生成画像を保存するプログラム
# args.load_model_path　は保存済みモデルへのパスです．このモデルを用いて，画像を生成します．
# args.save_generated_image_pathは，生成画像を保存したいディレクトリーへのパスです．
# 512枚の生成画像をランダムに生成します．
# 生成された画像と，実際の画像を用いて，FIDスコアを計算できます．
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from tqdm import tqdm
from model import Generator, Discriminator

manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def check_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p)

def arg_parser():
    parser = argparse.ArgumentParser()

    #generatorのパス
    parser.add_argument('--load_model_path', type=str, default="./result/model/gen_150.pt")
    #生成画像保存先のパス
    parser.add_argument('--save_generated_image_path', type=str, default="./generated_images/")
    #parser.add_argument('--save_image_path', type=str, default="./image/")
    #parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nz', type=int, default=100, help="Size of z latent vector (i.e. size of generator input)")
    parser.add_argument('--nc', type=int, default=3, help="Number of channels in the training images")
    parser.add_argument('--ngf', type=int, default=64, help="Size of feature maps in generator")
    parser.add_argument('--ndf', type=int, default=64, help="Size of feature maps in discriminator")
    parser.add_argument('--ngpu', type=int, default=1, help="Number of GPUs available. Use 0 for CPU mode")
    parser.add_argument('--workers', type=int, default=2, help="Number of workers for dataloader")

    args = parser.parse_args()
    return args

def test(args):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    PATH = args.load_model_path    
    Model = Generator(0, args.nc, args.nz, args.ngf)
    Model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
    Model.cuda()
    # print(Model)
    Model.eval()

    # fixed_noise = torch.randn(1, args.nz, 1, 1, device=device)
    # z_0_value = -2.0

    for n in range(10000):#作る画像枚数
        # fixed_noise[0, 45:50, 0, 0] = z_0_value　#潜在ベクトルを変更する場合
        noise = torch.randn(1, args.nz, 1, 1, device=device) #潜在ベクトルをランダムに作る場合
        fake = Model(noise).detach().cpu()
        vutils.save_image(fake,
                          os.path.join(args.save_generated_image_path, f"fake_iter_{n:04}.jpg"),
                          nrow=8, value_range=(-1.0, 1.0), normalize=True) # 引数のエラーが出ることがあるかも
        # z_0_value += 0.2　#潜在ベクトルを変更する場合


def main(args):
    # check_dir(args.load_model_path)
    check_dir(args.save_generated_image_path)
    test(args)


if __name__ == "__main__":
    args = arg_parser()
    main(args)
