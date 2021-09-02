import os
import sys
import time
import glob
import numpy as np
# import torch
import torch.utils.data  # 如果不用这个就会出现pycharm不识别data的问题
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import random
import torch.nn.functional as F
# import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import scipy.io as sio

from model_search import Network
from architect import Architect
from utils import cutout

import global_variable as glv
glv._init()

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--num_class', type=int, default=9, help='classes of HSI dataset')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=2, help='cutout length')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=3e-4, help='weight decay for arch encoding')
parser.add_argument('--Train', default=200, help='Train_num')
parser.add_argument('--Valid', default=100, help='Valid_num')
parser.add_argument('--num_cut', type=int, default=10, help='band cutout')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.manualSeed = random.randint(1, 10000)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./result/log_3D.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# read data
image_file = r'C:\Matlab练习\duogun\PaviaU.mat'
label_file = r'C:\Matlab练习\duogun\PaviaU_gt.mat'
# 在本py文件内的使用：定义跨模块全局变量，赋值
glv.set_value('image_file', image_file)
glv.set_value('label_file', label_file)

image = sio.loadmat(image_file)
# Pavia = image['Pavia']  # 原版
b = image.keys()
m = []
for i in b:
    m.append(i)
Pavia = image[m[-1]]

label = sio.loadmat(label_file)
# GroundTruth = label['groundtruth']  # 原版
GroundTruth = label['lbs2']

Pavia = (Pavia - np.min(Pavia)) / (np.max(Pavia) - np.min(Pavia))

[nRow, nColumn, nBand] = Pavia.shape

nTrain = args.Train
nValid = args.Valid
num_class = int(np.max(GroundTruth))
HSI_CLASSES = num_class

HalfWidth = 16
Wid = 2 * HalfWidth

[row, col] = GroundTruth.shape

NotZeroMask = np.zeros([row, col])
NotZeroMask[HalfWidth + 1: -1 - HalfWidth + 1, HalfWidth + 1: -1 - HalfWidth + 1] = 1
G = GroundTruth * NotZeroMask

[Row, Column] = np.nonzero(G)
nSample = np.size(Row)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(seed, cut):
    print(seed)
    args.cutout = cut
    np.random.seed(seed)
    RandPerm = np.random.permutation(nSample)
    if not torch.cuda.is_available():
        logging.warning('no gpu device available')
        sys.exit(1)

    args.cutout = cut
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.manualSeed)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # global 字典变量赋值
    glv.set_value('num_bands', nBand)  # (200, 102, 1, 1)
    model = Network(nBand, args.init_channels, HSI_CLASSES, args.layers, criterion)
    model = model.to(device)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs // 2, 0.5)

    architect = Architect(model, args)

    min_valid_loss = 100

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    for epoch in range(1, args.epochs + 1):

        # imdb = {}	# 原版
        # imdb['data'] = np.zeros([Wid, Wid, nBand, nTrain + nValid], dtype=np.float32)
        # imdb['Labels'] = np.zeros([nTrain + nValid], dtype=np.int64)
        # imdb['set'] = np.zeros([nTrain + nValid], dtype=np.int64)

        imdb = {'data': np.zeros([Wid, Wid, nBand, nTrain + nValid], dtype=np.float32),
                'Labels': np.zeros([nTrain + nValid], dtype=np.int64),
                'set': np.zeros([nTrain + nValid], dtype=np.int64)}

        for iSample in range(nTrain):

            yy = Pavia[
                 Row[RandPerm[iSample]] - HalfWidth: Row[RandPerm[iSample]] + HalfWidth,
                 Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth, :]
            if args.cutout:
                xx = cutout(yy, args.cutout_length, args.num_cut)

                imdb['data'][:, :, :, iSample] = xx
            else:
                imdb['data'][:, :, :, iSample] = yy

            imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

        for iSample in range(nValid):
            imdb['data'][:, :, :, iSample + nTrain] = Pavia[
                                                      Row[RandPerm[iSample + nTrain]] - HalfWidth: Row[RandPerm[iSample + nTrain]]
                                                      + HalfWidth,
                                                      Column[RandPerm[iSample + nTrain]] - HalfWidth: Column[RandPerm[iSample + nTrain]]
                                                      + HalfWidth, :]
            imdb['Labels'][iSample + nTrain] = G[Row[RandPerm[iSample + nTrain]],
                                                 Column[RandPerm[iSample + nTrain]]].astype(np.int64)
        # print('Data is OK.')
        imdb['Labels'] = imdb['Labels'] - 1

        imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nValid]))).astype(np.int64)

        train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
        valid_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)

        train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                  shuffle=True, pin_memory=True, num_workers=0)
        valid_queue = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                                  shuffle=True, pin_memory=True, num_workers=0)

        tic = time.time()

        # lr = scheduler.get_lr()[0]    # 原版
        lr = scheduler.get_last_lr()[0]
        # training
        train_acc, train_obj, tar, pre = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)

        # validation
        valid_acc, valid_obj, tar_v, pre_v = infer(valid_queue, model, criterion)
        scheduler.step()
        toc = time.time()

        logging.info('Epoch %03d: train_loss = %f, train_acc = %f, val_loss = %f, val_acc = %f, time = %f',
                     epoch, train_obj, train_acc, valid_obj, valid_acc, toc - tic)

        if valid_obj < min_valid_loss:
            genotype = model.genotype()
            min_valid_loss = valid_obj
            logging.info('genotype = %s', genotype)

    return genotype


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    tar = np.array([])
    pre = np.array([])

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        # global device
        input = input.to(device)
        target = target.to(device)

        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.to(device)
        target_search = target_search.to(device)

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, t, p = utils.accuracy(logits, target, topk=(1,))
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()
    tar = np.array([])
    pre = np.array([])
    global device
    for step, (input, target) in enumerate(valid_queue):
        input = input.to(device)
        target = target.to(device)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, t, p = utils.accuracy(logits, target, topk=(1,))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre


if __name__ == '__main__':
    genotype = main(seed=np.random.randint(low=0, high=10000, size=1), cut=True)

    str1 = 'Searched Neural Architecture:'
    print(str1)
    print(genotype)

    # 保存优化结果到日志
    logging.info(str1)
    log_format = '%(message)s'  # 优化结果中不用显示时间等无关信息
    fh.setFormatter(logging.Formatter(log_format))
    logging.info(genotype)

    # 将优化结果自动写入到genotypes.py文件的末尾
    now = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(time.time()))
    resultStr = '\n# ' + now + str1 + '\n' + 'HSI = ' + str(genotype) + '\n'

    with open('genotypes.py', 'a') as f:
        f.write(resultStr)
        f.close()

    # 将本次运行中所使用的全局变量: image_file, label_file, num_bands
    # 写入到 global_variable.py 的初始化函数 _init()的全局变量字典_global_dict = {}当中去
    # 这样的话，就能实现当test_HSI.py 单独运行时，全局变量字典_global_dict 将以
    # 最近一次所使用的值被初始化

    # 生成最新行new_line
    prefix = '    _global_dict = {'
    _global_dict = {'image_file': image_file, 'label_file': label_file, 'num_bands': glv.get_value('num_bands')}
    new_line = prefix.strip('{') + str(_global_dict)

    # 替换
    filepath = 'global_variable.py'
    with open(filepath, mode='r', encoding='utf-8') as f:  # 'utf-8'可确保中文不乱码;如果fname不存在，则会创建fname文件
        lines = f.readlines()

    with open(filepath, mode='w', encoding='utf-8') as ff:  # 此处只能是 w，以 w 模式打开文件时，文件内容全被清空
        for line in lines:
            if line.startswith(prefix):
                line = new_line
            ff.write(line)
    # ————————————————
    # 修改指定行的方法参考了
    # 原文链接：https://blog.csdn.net/qq_36072270/article/details/103496152																				   