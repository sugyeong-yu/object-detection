# YOLOV3
# Dataload
## parsing
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

## Dataset , DataLoader
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

2. DataLoader (torch.utils.data.DataLoader
- data 객체에서 데이터를 읽어올 수 있음. 
- 데이터셋의 설정을 바꾸고 싶을때 parameter의 변경으로 가능하다.
  1. dataset : torch.utils.data.Dataset객체를 사용하거나 __len__과 __getitem__을 가진 class객체를 사용해야한다.
  2. batch_size (default = 1): data를 한번에 몇개씩 가져올 것인가.
  3. shuffle : 데이터를 무작위로 가져올 것인가. ( True면 무작위 False면 순차적으로)
  4. num_workers (default = 0): data가 main process로 불러오는 것을 의미한다. (멀티프로세싱 개수)
  5. collate_fn : map-style 데이터셋에서 sample list를 batch단위로 바꾸기위해 필요한 기능이다. zero-padding이나 variable size데이터 등 data size를 맞추기위해 주로 사용한다.


