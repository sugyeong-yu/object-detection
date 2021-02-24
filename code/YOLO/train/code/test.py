import argparse
import csv
import os
import time

import torch
import torch.utils.data
import numpy as np
import tqdm
from models import yolov3
from utils.dataset.Dataset import *
from utils.utils import *

def evaluate(model,path,image_size,batch_size,num_workers,device):
    model.eval()

    dataset=Dataset(path,image_size,augment=False, multiscale=False)
    dataloader=torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=False,num_workers=num_workers,collate_fn=dataset.collate_fn)

    labels=[]

    for _, images,targets in tqdm.tqdm(dataloader,desc='Evaluate method', leave=False):
        if targets is None:
            continue

        # label추출
        labels.extend(targets[:,1].tolist()) # targets[:1]은 몇번쨰 class인지 번호
        targets[:, 2:] = corner(targets[:, 2:]) # x,y,w,h -> x1,y1,x2,y2
        targets[:, 2:] *= image_size # 1 * 1 -> img_size * img_size 기준으로 변환

