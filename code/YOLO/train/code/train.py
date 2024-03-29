import os, time
import torch
import torch.utils.data
import argparse
import torch.utils.tensorboard
from tqdm import trange, tqdm
from utils.utils import *
from utils.dataset.Dataset import *
import time
from models import yolov3

if __name__== "__main__":

    # Data parse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_config", type=str, default="E:\study\sugyeong_github\object-detection\code\YOLO\config\coco_data.cfg", help="path to data config file")
    parser.add_argument("--multiscale_training",type=bool,default=True,help="allow for multi-scale training")
    parser.add_argument("--image_size", type=int, default=416, help="size of each image")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--num_workers", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--gradient_accumulation",type=int,default=1,help="number of gradient accums before step")
    parser.add_argument("--epoch",type=int,default=100,help="number of epoch")
    parser.add_argument("--pretrained_weights", type=str, default='../../weights/darknet53.conv.74',help="if specified starts from checkpoint model") # weight불러오기

    args = parser.parse_args() # 저장
    print(args)

    # Tensorboard writer 객체 생성
    now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))
    log_dir = os.path.join('logs', now)
    os.makedirs(log_dir, exist_ok=True)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_config = parse_data_config(args.data_config) # txt경로들을 받아옴
    train_path = data_config['train']
    valid_path = data_config['valid']
    num_classes = int(data_config['classes'])
    class_names = load_classes(data_config['names'])

    dataset = Dataset(train_path,args.image_size,augment=False,multiscale=args.multiscale_training)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # model
    model = yolov3.Yolo_v3(args.image_size,num_classes).to(device)
    model.apply(init_weights_normal) # model.apply(f) > 현재 모듈의 모든 서브모듈에 해당함수f를 적용한다. (모델파라미터 초기화할때 많이사용)
    if args.pretrained_weights.endswith('.pth'):
        # args.pretrained_weights의 맨 뒷글자가 .pth이면 yolo학습된 파일 불러오기
        model.load_state_dict(torch.load(args.pretrained_weights))
    else:
        # darknet 학습된 파일 불러오기
        model.load_darknet_weights(args.pretrained_weights)

    #optimizer설정
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    schedular = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.8)

    # 현재 배치 손실값을 출력하는 tqdm설정
    loss_log = tqdm(total=0, position=2, bar_format= '{desc}',leave=True)
    #
    for epoch in tqdm(range(args.epoch), desc='Epoch'):
        model.train() # train mode
        for batch_idx, (_, images, targets) in enumerate(tqdm(dataloader, desc='Batch', leave=False)):
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

            for i, yolo_layer in enumerate(model.yolo_layers):
                print('loss_bbox_{}'.format(i + 1), yolo_layer.metrics['loss_bbox'], step)
                print('loss_conf_{}'.format(i + 1), yolo_layer.metrics['loss_conf'], step)
                print('loss_cls_{}'.format(i + 1), yolo_layer.metrics['loss_cls'], step)
                print('loss_layer_{}'.format(i + 1), yolo_layer.metrics['loss_layer'], step)
            print('total_loss', loss.item(), step)

        # 1개의 epoch 완료 후
        schedular.step() # lr_scheduler step 진행
    #     precision, recall, AP, f1, _,_,_ = test.evaluate()
    # # Tensorboard에 평가 결과 기록
    #     writer.add_scalar('val_precision', precision.mean(), epoch)
    #     writer.add_scalar('val_recall', recall.mean(), epoch)
    #     writer.add_scalar('val_mAP', AP.mean(), epoch)
    #     writer.add_scalar('val_f1', f1.mean(), epoch)
    #
    #     # checkpoint file 저장
    #     save_dir = os.path.join('checkpoints', now)
    #     os.makedirs(save_dir, exist_ok=True)
    #     dataset_name = os.path.split(args.data_config)[-1].split('.')[0]
    #     torch.save(model.state_dict(), os.path.join(save_dir, 'yolov3_{}_{}.pth'.format(dataset_name, epoch)))
