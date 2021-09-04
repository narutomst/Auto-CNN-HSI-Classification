import os
import os.path as osp
import sys
import time
import glob
import numpy as np
import torch.utils.data
import torch
from numpy.core._multiarray_umath import ndarray
import utils
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import scipy.io as sio

from model import NetworkHSI
from sklearn.metrics import confusion_matrix
from utils import cutout
from data_prepare import read_data, load_data
import global_variable as glv

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--num_class', type=int, default=9, help='classes of HSI dataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.016, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=3, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=2, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--arch', type=str, default='HSI', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--num_cut', type=int, default=10, help='band cutout')
parser.add_argument('--Train', type=int, default=200, help='Train_num')
parser.add_argument('--Valid', type=int, default=100, help='Valid_num')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.manualSeed = random.randint(1, 10000)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('./result/log_3D.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# read data
# image_file = r'C:\Matlab练习\duogun\PaviaU.mat'
# label_file = r'C:\Matlab练习\duogun\PaviaU_gt.mat'

# 本文件中的数据集地址不需要手动设置，由程序直接从global_variable.py模块中提取
# 以保证此处的地址与HSI_search.py中的一致。
glv._init()
image_file = glv.get_value('image_file')
label_file = glv.get_value('label_file')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(genotype, seed, cut=False):
    data, shuffle_number = read_data(image_file, label_file, train_nsamples=200, validation_nsamples=100, windowsize=32,
                                     istraining=True, shuffle_number=None, batchnumber=1000, times=0, rand_seed=seed)

    image, label = load_data(image_file, label_file)
    # 取得HSI数据尺寸
    [nRow, nColumn, nBand] = image.shape
    # 取得地物类别数量
    num_class = int(np.max(label))

    HalfWidth = 16
    Wid = 2 * HalfWidth
    [row, col] = label.shape

    NotZeroMask = np.zeros([row, col])
    NotZeroMask[HalfWidth + 1: -1 - HalfWidth + 1, HalfWidth + 1: -1 - HalfWidth + 1] = 1
    # NotZeroMask[17:-16, 17:-16] = 1, 负索引 i 的含义是从数组的末尾开始计数(
    # 即，如果i < 0 ，被解释为 n + i，其中 n 是相应维度中的元素数量
    # row=610, col=340, 则上面的切片表达式被解释为NotZeroMask[17:610-16, 17:340-16] = 1
    # 并且，numpy中的切片索引是计头不计尾，即i:j 表示i,i+1,...,(j-1)
    # 也就是说，整幅图片的上下左右四个方向上，边缘的16行、16列被去掉了。
    G = label * NotZeroMask  # 对应元素相乘 element-wise product: np.multiply(), 或 *
    # 返回G中非零元素的行索引和列索引值
    [Row, Column] = np.nonzero(G)
    # 统计整张HSI图片上的非零label的样本总数。
    # 将以下关键变量的名称与data_prepare中保持一致
    number_samples = np.size(Row)

    train_nsamples = args.Train
    validation_nsamples = args.Valid
    # total = train_nsamples + validation_nsamples
    test_nsamples = (number_samples - train_nsamples - validation_nsamples)

    batchtr = train_nsamples
    numbatch1 = train_nsamples // batchtr
    batchva = 1000
    numbatch2 = test_nsamples // batchva

    HSI_CLASSES = num_class
    # 散装语句到此结束

    # np.random.seed(rand_seed)
    # shuffle_number = np.random.permutation(number_samples)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    args.cutout = cut  # False
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.manualSeed)

    model = NetworkHSI(nBand, args.init_channels, HSI_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs // 5, 0.5)

    min_val_obj = 100
    for epoch in range(args.epochs):
        tic = time.time()
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        predict = np.array([], dtype=np.int64)
        labels = np.array([], dtype=np.int64)

        imdb = {'data': np.zeros([windowsize, windowsize, nBand, train_nsamples + validation_nsamples],
                                 dtype=np.float32),
                'Labels': np.zeros([train_nsamples + validation_nsamples], dtype=np.int64),
                'set': np.hstack((np.ones([train_nsamples]), 3 * np.ones([validation_nsamples]))).astype(np.int64)}
        for i in range(train_nsamples):
            c_row = Row[shuffle_number[i]]
            c_col = Column[shuffle_number[i]]
            yy = image[c_row - HalfWidth: c_row + HalfWidth,
                       c_col - HalfWidth: c_col + HalfWidth, :]
            if args.cutout:
                xx = cutout(yy, args.cutout_length, args.num_cut)
                imdb['data'][:, :, :, i] = xx
            else:
                imdb['data'][:, :, :, i] = yy

            imdb['Labels'][i] = G[c_row, c_col].astype(np.int64)

        for i in range(validation_nsamples):
            c_row = Row[shuffle_number[i + train_nsamples]]
            c_col = Column[shuffle_number[i + train_nsamples]]
            imdb['data'][:, :, :, i + train_nsamples] = image[c_row - HalfWidth:c_row + HalfWidth,
                                                              c_col - HalfWidth:c_col + HalfWidth, :]
            imdb['Labels'][i + train_nsamples] = G[c_row, c_col].astype(np.int64)
        imdb['Labels'] = imdb['Labels'] - 1

        train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
        valid_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)

        train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                  shuffle=True, num_workers=0)
        valid_queue = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=0)

        # training      tar:target  pre:predict
        train_acc, train_obj, tar, pre = train(train_queue, model, criterion, optimizer)
        # validation
        valid_acc, valid_obj, tar_v, pre_v = infer(valid_queue, model, criterion)
        scheduler.step()
        toc = time.time()

        logging.info('Epoch: %03d train_loss: %f train_acc: %f valid_loss: %f valid_acc: %f' % (
            epoch + 1, train_obj, train_acc, valid_obj, valid_acc))

        if epoch > args.epochs * 0.8 and valid_obj < min_val_obj:
            min_val_obj = valid_obj
            utils.save(model, './result/weights.pt')

    utils.load(model, './result/weights.pt')
    # matrix = test_model(model, shuffle_number, seed)

    return matrix


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()
    tar = np.array([])
    pre = np.array([])

    for step, (input, target) in enumerate(train_queue):
        # global device
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, t, p = utils.accuracy(logits, target, topk=(1,))
        n = input.size(0)
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
    # global device
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


def test_model(model, numbatch2, seed):
    model.eval()
    total_tar = np.array([])
    total_pre = np.array([])

    i = 0
    test_nsamples = 0

    # test
    # if epoch == args.epochs:
    # utils.load(model, './result/weights.pt')
    for i in range(numbatch2):
        global windowsize, HalfWidth, nBand, batchva, criterion, image, Row, shuffle_number, train_nsamples, validation_nsamples, Column, G
        imdb = {'data': np.zeros([windowsize, windowsize, nBand, batchva], dtype=np.float32),
                'Labels': np.zeros([batchva], dtype=np.int64),
                'set': 3 * np.ones([batchva], dtype=np.int64)}
        for j in range(batchva):
            c_row = Row[shuffle_number[j + train_nsamples + validation_nsamples + i * batchva]]
            c_col = Column[shuffle_number[j + train_nsamples + validation_nsamples + i * batchva]]
            imdb['data'][:, :, :, j] = image[c_row - HalfWidth:c_row + HalfWidth,
                                             c_col - HalfWidth:c_col + HalfWidth, :]
            imdb['Labels'][j] = G[c_row, c_col].astype(np.int64)

        imdb['Labels'] = imdb['Labels'] - 1

        test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)

        test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=50,
                                                 shuffle=False, num_workers=0)

        valid_acc, valid_obj, tar_v, pre_v = infer(test_queue, model, criterion)

        predict = np.append(predict, pre_v)
        labels = np.append(labels, tar_v)

    OA_V = sum(map(lambda x, y: 1 if x == y else 0, predict, labels)) / (numbatch2 * batchva)
    matrix = confusion_matrix(labels, predict)

    logging.info('test_acc= %f' % (OA_V))
    return matrix


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    # AA = np.zeros([shape[0]], dtype=np.float)
    TPR: ndarray = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        TPR[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(TPR)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA, Kappa, TPR


if __name__ == '__main__':
    genotype = eval('genotypes.{}'.format(args.arch))
    matrix = main(genotype=genotype, seed=np.random.randint(low=0, high=10000, size=1), cut=False)
    OA, AA, Kappa, TPR = cal_results(matrix)
    print(OA)

    # 将分类结果按列保存到txt中
    now = time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime(time.time()))
    resultStr = '# OA, AA, Kappa, TPR  ' + now + image_file + '\n'

    # 不同数据集的分类结果保存到不同的文件中，fname
    dir_name, base_name = osp.split(image_file)
    f_name, f_ext = osp.splitext(base_name)
    fname = './result/classification_3D_' + f_name + '.txt'

    a = list([OA, AA, Kappa])
    a.extend(list(TPR))
    # X = np.array(a)
    # 使用f和np.savetxt()组合来保存数据，格式处理很方便
    # 数据的标题只写一次，只要不是空文件，追加的时候就不用再写标题了
    # 数据文件fname不存在的话，open()就会自动创建fname文件
    with open(fname, mode='a', encoding='utf-8') as f:  # 'utf-8'可确保中文不乱码;如果fname不存在，则会创建fname文件
        if os.path.getsize(fname) == 0:
            header = resultStr.rstrip('\n')
        else:
            header = ''
        np.savetxt(f, np.column_stack(a), fmt='%.4f', delimiter=' ', newline='\n', header=header,
                   footer='', comments='', encoding=None)
        f.close()

    OA, AA_mean, Kappa, AA = cal_results(matrix)
    print(OA)
