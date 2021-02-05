# YOLOV3
# Dataload
## train.py 관련 함수 및 모듈
### parsing
```
import argparse

parser = argparse.ArgumentParser() # 1
parser.add_argument("--epoch", type=int, default=100, help="number of epoch") # 2
parser.add_argument("--gradient_accumulation", type=int, default=1, help="number of gradient accums before step")

args = parser.parse_args() # 3
print(args.epoch) # 4
print(args.gradient_accumulation)
```
- 1: 인자 값을 받을 수 있는 인스턴스 생성
- 2: 입력받을 인자값 등록
- 3: 입력받은 인자값을 args에 저장 (type = namespace)
- 4: 입력받은 인자값 출력

1. argparse.ArgumentParser()\
![image](https://user-images.githubusercontent.com/70633080/106458577-f074de00-64d3-11eb-9fff-c8ea6de6a96e.png)

2. add_argument()\
![image](https://user-images.githubusercontent.com/70633080/106458624-01255400-64d4-11eb-9a89-a41355b7fc13.png)

### Dataset , DataLoader
- Pytorch의 Dataset과 DataLoader를 사용하면 학습을 위한 방대한 데이터를 미니배치단위로 처리할 수 있고 데이터를 무작위로 섞음으로서 학습효율성을 향상시킬 수 있다.
```
dataset = utils.datasets.ListDataset(train_path, args.image_size, augment=True, multiscale=args.multiscale_training)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         pin_memory=True,
                                         collate_fn=dataset.collate_fn)
```
1. Dataset (torch.utils.data.Dataset)
- torch의 dataset은 2가지 style이 있다.
  1. Map-style dataset
    - index가 존재하여 data[index]로 데이터참조가 가능하다.
    - __getitem__과 __len__선언이 필요하다.
  2. Iterable-style dataset
    - random으로 읽기에 어렵거나 data에 따라 batch size가 달라지는 data에 적합하다.
    - 비교하자면 stream data, real-time log에 적합하다
    - __iter__선언이 필요하다.
- 본 코드에서는 직접 Dataset 객체를 만들었다. (__len__과 __getitem__을 구현하면 DataLoader로 데이터를 불러 올 수 있다.)
- __len__ : 데이터의 크기를 반환
- __getitem__ : index로 data를 return해주는 class

2. DataLoader (torch.utils.data.DataLoader)
- data 객체에서 데이터를 읽어올 수 있음. 
- 데이터셋의 설정을 바꾸고 싶을때 parameter의 변경으로 가능하다.
  1. dataset : torch.utils.data.Dataset객체를 사용하거나 __len__과 __getitem__을 가진 class객체를 사용해야한다.
  2. batch_size (default = 1): data를 한번에 몇개씩 가져올 것인가.
  3. shuffle : 데이터를 무작위로 가져올 것인가. ( True면 무작위 False면 순차적으로)
  4. num_workers (default = 0): data가 main process로 불러오는 것을 의미한다. (멀티프로세싱 개수)
  5. collate_fn : map-style 데이터셋에서 sample list를 batch단위로 바꾸기위해 필요한 기능이다. zero-padding이나 variable size데이터 등 data size를 맞추기위해 주로 사용한다.

## Dataset.py 관련 함수 및 모듈
----------------------------------------------------------------------------------
### Class Dataset - __getitem__() 
#### Transforms
- torchvision.transforms.Compose()\
: Compose()는 여러 transform들을 compose로 구성할 수 있다.
```
transforms = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(brightness=1.5, saturation=1.5, hue=0.1),
                torchvision.transforms.ToTensor()
            ])
```
- 사용할 수 있는 함수
  - transforms.ToPILImage() : csv파일로 데이터셋을 받은 경우, PIL Image로 바꿔줌
  - transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0) : 이미지의 밝기, 대비 및 채도를 임의로 변경한다.
  - transforms.CenterCrop(size) : 가운데 부분을 size크기로 자른다.
  - transforms.Grayscale(num_output_channels=1) : grayscale로 변환한다.
  - transforms.RandomAffine(degrees) : 랜덤으로 affine변형을 한다.
  - transforms.RandomCrop(size) : 이미지를 랜덤으로 아무데나 잘라 size크기로 출력한다.
  - transforms.Resize(size) : 이미지사이즈를 size로 변경한다.
  - transforms.RandomRotation(degrees) : 이미지를 랜덤으로 degress각도로 회전한다.
  - transforms.RandomResizedCrop(size, scale=(0.08,1.0), ratio=(0.75, 1.3333333)) : 이미지를 랜덤으로 변형한다.
  - transforms.RandomVerticalFlip(p=0.5) - 이미지를 랜덤으로 수직으로 뒤집는다. p =0이면 뒤집지 않는다.
  - transforms.RandomHorizontalFlip(p=0.5) - 이미지를 랜덤으로 수평으로 뒤집는다.
  - transforms.ToTensor() - 이미지 데이터를 tensor로 바꿔준다.
  - transforms.Normalize(mean, std, inplace=False) - 이미지를 정규화한다.
  
 #### img 정사각형으로 만들기
 ```
 def pad_to_square(image, pad_value=0):
    _, h, w = image.shape
    # 너비와 높이의 차
    difference = abs(h - w)

    # (top, bottom) padding or (left, right) padding
    if h <= w:
        top = difference // 2
        bottom = difference - difference // 2
        pad = [0, 0, top, bottom]
    else:
        left = difference // 2
        right = difference - difference // 2
        pad = [left, right, 0, 0]

    # Add padding
    image = F.pad(image, pad, mode='constant', value=pad_value)
    return image, pad
```
- 가로가 세로보다 긴 경우, 세로가 가로보다 긴 경우를 나누어 padding을 처리해준다.\
<img src="https://user-images.githubusercontent.com/70633080/106993249-a6367a00-67bd-11eb-83ea-12395faf127d.png" width=70% height=70%>\
<img src="https://user-images.githubusercontent.com/70633080/106993286-bbaba400-67bd-11eb-8234-bb438bd2e580.png" width=70% height=70%>
- F.pad(img, pad=(n,m), value) : img의 앞에 n개, 뒤에 m개의 value를 padding하여 반환한다. 

#### labeling
- np.loadtxt(txt_path) : txt를 한줄한줄 읽어옴. 
- 패딩 및 스케일링 전의 이미지크기에 맞게 anchor box size와 위치조정
```
x1 = w * (boxes[:, 1] - boxes[:, 3] / 2)  # 앵커박스의 좌상단x좌표 * 이미지w크기
y1 = h * (boxes[:, 2] - boxes[:, 4] / 2)  # 앵커박스의 좌상단y좌표 * 이미지h크기
x2 = w * (boxes[:, 1] + boxes[:, 3] / 2)  # 앵커박스의 우하단x좌표 * w크기
y2 = h * (boxes[:, 2] + boxes[:, 4] / 2)  # 앵커박스의 우하단y좌표 * h크기
```
- 이는 아래그림처럼 기존 이미지에서의 anchor box의 좌상단, 우하단 좌표를 구하는 과정이다.\
<img src="https://user-images.githubusercontent.com/70633080/106993816-e518ff80-67be-11eb-82fe-a9910b6d2d38.png" width=80% height=80%>

---------------------------------------------------------------------

## 참고
- dataset.py - <https://github.com/lulindev/yolov3-pytorch/blob/df89032d6764ba09524445d17007561aad9a1ea1/utils/datasets.py#L40>
- cocodataset.py - <https://velog.io/@dkdk6638/Pytorch-COCO-Dataset>
