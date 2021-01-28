import os, time
import torch
import torch.utils.data
import torch.utils.tensorboard
import tqdm
import yolov3



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
model = yolov3.Yolo_v3().to(device)
#model.apply(utils,utils.init_weights_normal) # model.apply(f) > 현재 모듈의 모든 서브모듈에 해당함수f를 적용한다. (모델파라미터 초기화할때 많이사용)


#optimizer설정
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
# lr schedular설정
# 1. LambdaLR : lr_lambda인자로 넣어준 함수로 계산된 값을 초기 lr에 곱해 사용
# 2. MultiplicativeLR : lr_lambda 인자로 넣어준 함수로 계산된 값을 매 에폭마다 이전 lr에 곱해사용한다.
# 3. StepLR : step_size에 지정된 에폭수마다 이전 lr에 감마만큼 곱해 사용한다.
# 4. ExponentialLR : 매 에폭마다 이전 lr에 감마만큼 곱해서 사용한다.
schedular = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.8)

# 현재 배치 손실값을 출력하는 tqdm설정
# tqdm은 작업시간이 얼마나 남았는지 확인하고 싶을때 진행상태바를 만들 수 있는 라이브러리이다.
# from tqdm import tqdm
# total : 전체 반복량, bar_format : str , leave : bool, default로 True (진행상태에서 잔상이남음)
loss_log = tqdm.tqdm(total=0, position=2, bar_format= '{desc}',leave=False)