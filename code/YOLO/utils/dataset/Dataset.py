import torch, torchvision
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import random


def parse_data_config(path: str):
    """데이터셋 설정 파일을 parse한다."""
    options = {}
    with open(path, 'r') as f:
        lines = f.readlines()  # lines 는 train.txt경로 valid.txt경로
    for line in lines:
        print(line)
        line = line.strip()  # strip() > 문자열의 양끝에 존재하는 공백과 \n을 제거해줌
        key, value = line.split('=')  # key는 train value는 train.txt경로
        options[key.strip()] = value.strip()  # dict반환
    return options

def load_classes(path: str):
    """클래스 이름을 로드한다."""
    with open(path, "r") as f:
        names = f.readlines()
    for i, name in enumerate(names):
        names[i] = name.strip()
    return names


def pad_to_square(image, pad_value=0):
    _, h, w = image.shape

    diff = abs(h - w)

    if h <= w:
        top = diff // 2
        bottom = diff - diff // 2
        pad = [0, 0, top, bottom]  # 한 열에 대해 각각 앞 뒤 를 top, bottom 개수만큼 채워준다. ex) (1,1) -> [(0,0),(1,1),(0,0)]
    else:
        left = difference // 2
        right = diff - diff // 2
        pad = [left, right, 0, 0]  # 한 행에 대해 각각 앞 뒤 를 left, right 개수만큼 채워준다. ex) (1,1) -> (0,1,1,0)

    image = F.pad(image, pad, mode='constant', value=pad_value)
    return image, pad

def resize(image,size):
    return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, datapath: str, image_size: int, augment: bool, multiscale :bool ):
        # 경로가 주어지면 바로 그 경로안의 이미지들을 읽을수도 있지않을까?
        with open(datapath, 'r') as file:
            self.img_files = file.readlines()  # text파일에서 image하나당 경로를 읽어옴.

        # path ./data/coco/images/val2014/COCO_val2014_000000580607.jpg\n' ->./data/coco/labels/val2014/COCO_val2014_000000581736.txt\n'
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                                .replace('JPEGImages', 'labels') for path in self.img_files]

        self.image_size = image_size
        self.max_objects = 100
        self.augment = augment  # 어그먼테이션 (이미지변형)
        self.multiscale = multiscale
        # self.normalized_labels = normalized_labels
        self.batch_count = 0

    def __getitem__(self, index):
        # 1. image처리
        image_path = self.img_files[index].rstrip() # rstrip()은 문자열의 지정된 문자열의 끝을 (기본값은 공백) 삭제
        # img augmentation
        if self.augment:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.1),
                torchvision.transforms.ToTensor()
            ])
        else:
            transforms = torchvision.transforms.ToTensor()

        # tensor에서 img추출하기
        image = transforms(Image.open(image_path).convert('RGB'))  # IMG는 pillow에서 제공. convert로 RGB로 전환
        _, h, w = image.shape  # torch.Size([3, 480, 640])

        # img 정사각형?으로 만들어주기
        image, pad = pad_to_square(image)
        _, pad_h, pad_w = image.shape

        # # 2. label처리
        label_path = self.label_files[index].rstrip()

        targets = None
        if os.path.exists(label_path):
            # np.loadtxt : ("파일경로",파일구분자,데이터타입) 을 지정하여 파일을 읽어와 데이터변수에 array로 넣어준다.
            # 8*(class, x,y,w,h) 형태로 반환
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))  # 5열의 형태로 만든후 array -> tensor

            # 패딩 및 스케일링 전의 이미지크기에 맞게 앵커박스크기와 위치를 조정해준다.
            # ground truth의 앵커박스의 크기와 좌표는 grid size 1일때의 기준이다.
            x1 = w * (boxes[:, 1] - boxes[:, 3] / 2)  # 앵커박스의 좌상단x좌표 * 이미지w크기
            y1 = h * (boxes[:, 2] - boxes[:, 4] / 2)  # 앵커박스의 좌상단y좌표 * 이미지h크기
            x2 = w * (boxes[:, 1] + boxes[:, 3] / 2)  # 앵커박스의 우하단x좌표 * w크기
            y2 = h * (boxes[:, 2] + boxes[:, 4] / 2)  # 앵커박스의 우하단y좌표 * h크기

            # 앵커박스에도 padding해주기
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / pad_w
            boxes[:, 2] = ((y1 + y2) / 2) / pad_h
            boxes[:, 3] *= w / pad_w
            boxes[:, 4] *= h / pad_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes  # targets[:,0:1] 은 0으로 채워져있음.(cllate_fn에서 인덱스부여) 나머지는 class, x, y, w, h
            # print(targets)  # 8,6

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                image, targets = horisontal_flip(image, targets)
        # print(targets)
        return image_path, image, targets

    def __len__(self):
        return len(self.img_files)

    def collate_fn(self, batch):
        # __getitem__의 반환을 입력으로 받아옴 targets는 (0,class,x,y,w,h)*8
        # batch_size만큼의 path, img, target(bbox)를 받아온게 batch이다.
        paths, images, targets = list(zip(*batch)) # list(zip(data1,data2,data3...datan)

        # boxes가 없는 target은 지움.
        targets = [boxes for boxes in targets if boxes is not None]

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            # index로 접근하여 원소별로 값을 변경하면 기존의 값도 같이 변경된다. (주소값참조해서 바꾸는거라서 그러는거 같음)
            boxes[:, 0] = i # target마다 인덱스를 매김 첫번째target의 bbox index > 0 두번쨰target의 bbox index >1

        try:
            targets = torch.cat(targets, 0) # batch만큼의 target들을 하나의 tensor로 합침 (행기준으로)
        except RuntimeError:
            targets = None  # No boxes for an image

        # Selects new image size every 10 batches
        if self.multiscale and self.batch_count % 10 == 0:
            # yolo는 320부터 608까지의 다양한 scale로 resize를 하여 학습시킨다. (근데 bbox정보는 조정안해도되나?)
            self.image_size = random.choice(range(320, 608 + 1, 32)) # choice는 아무원소나 하나 뽑아줌. ()안에 있는것 중에.

        # Resize images to input shape
        images = torch.stack([resize(image, self.image_size) for image in images]) # img를 random으로 받아온 size로 resizing하여 쌓는다.
        self.batch_count += 1
        # return : (img의 경로, resize한 img, resize하기전 img에 대한 anchor box정보들)
        return paths, images, targets

# path = "../../data/coco/train.txt"
# batch1 = Dataset(path, 416, False,False).__getitem__(1)
# batch2 = Dataset(path, 416, False).__getitem__(2)
# path,img,targets= list(zip(batch1,batch2))
# print(path)

# targets = [boxes for boxes in targets if boxes is not None]
# #print(targets)
# for i, boxes in enumerate(targets):
#     boxes[:, 0] = i
#     print("boxes",boxes)
# targets = torch.cat(targets, 0)
# print(targets)
