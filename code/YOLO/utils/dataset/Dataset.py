import torch, torchvision
import torch.nn.functional as F
from PIL import Image
import os
import numpy as np
import random

def init_weights_normal(m):
    """정규분포 형태로 가중치를 초기화한다."""
    classname = m.__class__.__name__
    #print(classname) # Conv2d, BatchNorm2d, LeakyReLU 등...
    if classname.find("Conv2d") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, 0.1)

    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02) # model의 weight의 data를 평균이 1.0이고 표준편차가 0.02안 가우시안분포에 따라 tensor를 채움
        torch.nn.init.constant_(m.bias.data, 0.0) # 모델의 편향? 값을 0으로 채움.


def parse_data_config(path: str):
    """데이터셋 설정 파일을 parse한다."""
    options = {}
    with open(path, 'r') as f:
        lines = f.readlines()  # lines 는 train.txt경로 valid.txt경로
    for line in lines:
        #print(line)
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
        left = diff // 2
        right = diff - diff // 2
        pad = [left, right, 0, 0]  # 한 행에 대해 각각 앞 뒤 를 left, right 개수만큼 채워준다. ex) (1,1) -> (0,1,1,0)

    image = F.pad(image, pad, mode='constant', value=pad_value)
    return image, pad

def resize(image,size):
    return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_path: str, image_size: int, augment: bool, multiscale: bool, normalized_labels=True):
        with open(list_path, 'r') as file:
            self.image_files = file.readlines()

        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
                                .replace('JPEGImages', 'labels') for path in self.image_files]
        self.image_size = image_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.batch_count = 0

    def __getitem__(self, index):
        # 1. Image
        # -----------------------------------------------------------------------------------
        image_path = self.image_files[index].rstrip()

        # Apply augmentations
        if self.augment:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.1),
                torchvision.transforms.ToTensor()
            ])
        else:
            transforms = torchvision.transforms.ToTensor()

        # Extract image as PyTorch tensor
        image = transforms(Image.open(image_path).convert('RGB'))

        _, h, w = image.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        print("image_shape: ", h_factor,w_factor)
        # Pad to square resolution
        image, pad = pad_to_square(image)
        _, padded_h, padded_w = image.shape
        print("square: ", padded_h,padded_w)
        # 2. Label
        # -----------------------------------------------------------------------------------
        label_path = self.label_files[index].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            #print("before: ", boxes) > tensor([[23.0000,  0.7703,  0.4897,  0.3359,  0.6976],[23.0000,  0.1860,  0.9016,  0.2063,  0.1296]]
            # Extract coordinates for unpadded + unscaled image
            # 기존 460 * 640크기의 이미지에서의 좌표구하기(좌상단, 우하단)
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

            # Adjust for added padding
            # 640 * 640 크기의 이미지로 맞춰줌 padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            # Returns (x, y, w, h)
            # 1 * 1 크기의 이미지에서의 좌표로 변환. (정사각형, 패딩을 진행한것)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
            #print("target ", targets) # tensor([[ 0.0000, 23.0000,  0.7703,  0.4931,  0.3359,  0.4643],[ 0.0000, 23.0000,  0.1860,  0.7673,  0.2063,  0.0862]])
        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                image, targets = horisontal_flip(image, targets)

        return image_path, image, targets

    def __len__(self):
        return len(self.image_files)

    def collate_fn(self, batch):
        paths, images, targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        try:
            targets = torch.cat(targets, 0)
        except RuntimeError:
            targets = None  # No boxes for an image

        # Selects new image size every 10 batches
        if self.multiscale and self.batch_count % 10 == 0:
            self.image_size = random.choice(range(320, 608 + 1, 32))

        # Resize images to input shape
        images = torch.stack([resize(image, self.image_size) for image in images])
        self.batch_count += 1

        return paths, images, targets

path = "../../data/coco/train.txt"
batch1 = Dataset(path, 416, False,False).__getitem__(1)
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
