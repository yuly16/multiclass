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

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=10)
    parser.add_argument('--epoch1', type=int, default=10)
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
    optimizer1 = torch.optim.Adam(net.last_linear1.parameters(),lr=1e-4)

    # training

    # for i,j in train_dataloader:
    #     print(i.shape)
    #     print(j.shape)
    # train_dataset = Pascal_dataset(opt,mode='train')
        # test_dataset = Pascal_dataset(opt, mode='test')
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=opt['batch_size'], shuffle=True)

    for epoch in range(opt['epoch1']):
        net.train()
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
        acc = 0
        total = 0
        net.eval()
        for imgs, labels in tqdm(test_dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            y_pre = net(imgs)
            total = total+torch.sum(labels.squeeze())
            #print(y_pre[0])
            y_pre = torch.round(y_pre)
            zero = torch.zeros(y_pre.shape)
            zero = zero.to(device)
            M_and = torch.where(y_pre>0.5,labels,zero)
            acc += torch.sum(M_and)


        print("accuracy  %f"%(acc/total))




    # for imgs, labels in test_dataloader:
    #     imgs = imgs.to(device)
    #     labels = labels.to(device)
    #     print(len(test_dataloader))
if __name__ == "__main__":
    main()
