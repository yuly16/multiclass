import argparse
import torch
import numpy as np
import math
import random
from torch.utils.data import DataLoader
from pretrainedmodels import pretrainedmodels
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import os
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from inceptionresnetv2 import *
import numpy as np
#import cv2
import json
from tqdm import tqdm
#plt.switch_backend('agg')
# %matplotlib inline

from glob import glob
from tqdm import tqdm

#import cv2
from PIL import Image
import argparse
import torch
import numpy as np
import math

from torch.utils.data import DataLoader


global net

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + 1e-6)
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 1e-6)
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculate the F score, the weighted harmonic mean of precision and recall
    if beta < 0:
        raise ValueError('Lowest is zero')

    # If there are no true positives fix the F score at 0 like sklearn
    if torch.sum(torch.round(torch.clamp(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + 1e-6)
    return fbeta_score


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall
    return fbeta_score(y_true, y_pred, beta=1)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=6)
    parser.add_argument('--epoch1', type=int, default=20)
    parser.add_argument('--epoch2', type=int, default=8)
    parser.add_argument('--image_size', default=(299,299))
    parser.add_argument('--path',type=str,default='/home/yuly/multiclass/PascalVOC')
    parser.add_argument('--classes_num', type=int, default=20)
    opt=vars(parser.parse_args())

    #cuda config
    use_cuda=True if torch.cuda.is_available() else False
    #use_cuda = False
    device=torch.device('cuda:0') if use_cuda else torch.device('cpu')
    torch.manual_seed(1)
    if use_cuda:
        torch.cuda.manual_seed(1)
    img_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    train_dataset = pretrainedmodels.datasets.Voc2007Classification(opt['path'], 'train', transform=img_transform)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=opt['batch_size'],shuffle=True)
    test_dataset = pretrainedmodels.datasets.Voc2007Classification(opt['path'], 'test', transform=img_transform)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=opt['batch_size'],shuffle=True)
    net = inceptionresnetv2(num_classes=20, pretrained='imagenet')
    net.to(device)
    #print([name for name,x in net.named_parameters()])
    optimizer1 = torch.optim.Adam(list(net.last_linear1.parameters())+list(net.old_module.last_linear.parameters()),lr=1e-3)

    # training

    # for i,j in train_dataloader:
    #     print(i.shape)
    #     print(j.shape)
    # train_dataset = Pascal_dataset(opt,mode='train')
        # test_dataset = Pascal_dataset(opt, mode='test')
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=opt['batch_size'], shuffle=True)

    max_f1 = 0
    for epoch in range(opt['epoch1']):
        net.train()
        print('training procedure:')
        for imgs, labels in tqdm(train_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer1.zero_grad()
            criterion = torch.nn.BCELoss()
            y_pre = net(imgs)
            #print(y_pre.shape)
            BCE_loss = criterion(y_pre,labels)
            BCE_loss.backward()
            optimizer1.step()
        print("loss %f the %d step in total %d epochs finished"%(BCE_loss,epoch,opt['epoch1']))
        # test
        print('\n')
        print('testing procedure:')
        f1 = []
        pre = []
        rec = []
        total = 0
        net.eval()
        for imgs, labels in tqdm(test_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            y_pre = net(imgs)

            # total = total+torch.sum(labels.squeeze())
            # #print(y_pre[0])
            # y_pre = torch.round(y_pre)
            # zero = torch.zeros(y_pre.shape)
            # zero = zero.to(device)
            # M_and = torch.where(y_pre>0.5,labels,zero)
            # acc += torch.sum(M_and)
            f1.append(fmeasure(labels,y_pre))
            pre.append(precision(labels,y_pre))
            rec.append(recall(labels,y_pre))
        mf1 = np.mean(f1)
        print("f1-score: %f  precision: %f   recall: %f "%(mf1,np.mean(pre),np.mean(rec)))
        if mf1>max_f1:
            torch.save(net,'best_model1.pth')
            print('best model, saving…')



        print("accuracy  %f"%(acc/total))




    # for imgs, labels in test_dataloader:
    #     imgs = imgs.to(device)
    #     labels = labels.to(device)
    #     print(len(test_dataloader))
if __name__ == "__main__":
    main()
