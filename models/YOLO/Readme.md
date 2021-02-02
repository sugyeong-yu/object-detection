# YOLOV3
# Dataload
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

