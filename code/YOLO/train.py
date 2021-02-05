import os, time
import torch
import torch.utils.data
import argparse
import torch.utils.tensorboard
from tqdm import trange, tqdm
import utils.utils
import utils.Dataset
import time
from models import yolov3


# Data parse
parser = argparse.ArgumentParser()

parser.add_argument("--data_config", type=str, default="E:\study\sugyeong_github\object-detection\models\YOLO\config\coco.cfg", help="path to data config file")
parser.add_argument("multiscale_training",type=bool,default=True,help="allow for multi-scale training")
parser.add_argument("--image_size", type=int, default=416, help="size of each image")
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--num_workers", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("gradient_accumulation",type=int,default=1,help="number of gradient accums before step")
# parser.add_argument("epoch",type=int,default=100,help="number of epoch")
# parser.add_argument("--pretrained_weights", type=str, default='weights/darknet53.conv.74',help="if specified starts from checkpoint model") # weight불러오기

args = parser.parse_args() # 저장
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_config = utils.parse_data_config(args.data_config) # txt경로들을 받아옴
train_path = data_config['train']
valid_path = data_config['valid']

dataset = utils.Dataset.Dataset(train_path,args.image_size,augment=False,multiscale=args.multiscale_traning)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         pin_memory=True,
                                         collate_fn=dataset.collate_fn)

# model
model = yolov3.Yolo_v3().to(device)
#model.apply(utils,utils.init_weights_normal) # model.apply(f) > 현재 모듈의 모든 서브모듈에 해당함수f를 적용한다. (모델파라미터 초기화할때 많이사용)
# if args.pretrained_weights.endswith('.pth'):
#     model.load_state_dict(torch.load(args.pretrained_weights))
# else:
#     model.load_darknet_weights(args.pretrained_weights)

#optimizer설정
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

schedular = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.8)

# 현재 배치 손실값을 출력하는 tqdm설정
# tqdm은 작업시간이 얼마나 남았는지 확인하고 싶을때 진행상태바를 만들 수 있는 라이브러리이다.
# total : 전체 반복량, bar_format : str , leave : bool, default로 True (진행상태에서 잔상이남음)
loss_log = tqdm(total=0, position=1, bar_format= '{desc}',leave=True)

# for i in tqdm(range(10)):
#     time.sleep(0.1)
#     loss_log.set_description_str('Loss: {:.6f}'.format(1)) # 진행바의 이름을 바꿔줄 수 있음

for epoch in tqdm.tqdm(range(args.epoch), desc='Epoch'):
    model.train() # train mode
    for batch_idx, (_, images, targets) in enumerate(tqdm.tqdm(dataloader, desc='Batch', leave=False)):
        step = len(dataloader) * epoch + batch_idx

        images = images.to(device)
        targets = targets.to(device)

        # forward, backward
        loss, outputs = model(images, targets)
        loss.backward()

        if step % args.gradient_accumulation == 0:
            # step마다 기울기 저장.
            optimizer.step()
            optimizer.zero_grad()

        # 총 loss값 계산
        loss_log.set_description('Loss: {:.6f}'.format(loss.item()))

    # 1개의 epoch 완료 후
    schedular.step() # lr_scheduler step 진행
    precision, recall, AP, f1, _,_,_ = test.evaluate()

