# R-CNN
R-CNN이란?\
Image classification을 수행하는 CNN과 localization을 위한 regional proposal알고리즘을 연결한 모델
- Region Proposal - CNN - SVM 
![image](https://user-images.githubusercontent.com/70633080/102708762-3e4cd280-42e8-11eb-82ae-273588515824.png)
- R-CNN의 Object detection 알고리즘
1. 입력이미지에 selective search 알고리즘을 적용해 물체가 있을만한 박스 2000개 추출
2. 모든 박스를 227*227 사이즈로 resize. (박스의 비율은 고려하지 않는다.)
> Convolution layer에는 input size가 고정이지 않지만 마지막 FC layer에서의 input size가 고정이므로 Convolution에 대한 output size가 동일해야하기 때문이다.
3. 이미지넷 데이터를 통해 학습된 CNN을 통과시켜 4096차원의 특징벡터를 추출.
4. 추출된 벡터로 각 클래스마다 학습시켜놓은 SVM Classifier를 통과.
5. 바운딩 박스 리그레션을 적용해 박스의 위치를 조정

## 1. Region Proposal
> Region Proposal이란 주어진 이미지에서 물체가 있을법한 위치를 찾는것이다.\
> 기존의 sliding window 방식을 보완한것. 
> ### sliding window
> > 이미지에서 물체를 찾기위해 window의 크기,비율을 임의로 바꿔가며 모든영역에 대해 탐색하는것\
> > ![image](https://user-images.githubusercontent.com/70633080/102751575-64877680-43ab-11eb-805b-b4087ec78a36.png)
> > - 모든 영역을 탐색하기에는 너무 느리다.
> > - 비효율적
> > 따라서 R-CNN에서는 이를 극복하기 위해 다른 알고리즘을 사용한다.\
> R-cnn은 " Selective Search "라는 룰베이스 알고리즘을 통해 2000개의 물체박스를 찾는다.
> ### Selective Search
> > 주변 픽셀 간의 유사도를 기준으로 Segmentation을 만들고, 이를 기준으로 물체가 있을법한 박스를 추론\
> > ![image](https://user-images.githubusercontent.com/70633080/102708835-c3d08280-42e8-11eb-872e-e4af63ccfb51.png)
> - 그러나 R-CNN이후 region proposal 과정은 뉴럴 네트워크가 수행하도록 발전되었다. 따라 더이상 사용하지 않는 알고리즘이다.
## 2. Feature Extraction
> Selective Search를 통해서 찾아낸 2천개의 박스 영역은 227 x 227 크기로 리사이즈 됩니다.\
> 그리고 Image Classification으로 미리 학습되어 있는 CNN 모델을 통과하여 4096 크기의 특징 벡터를 추출한다.
> - 미리학습된 모델이란?\
> 이미지넷 데이터(ILSVRC2012 classification)로 미리 학습된 CNN 모델을 가져온 다음, fine tune하는 방식/
>  Classification의 마지막 레이어를 Object Detection의 클래스 수 N과 아무 물체도 없는 배경까지 포함한 N+1로.
> - 미리 이미지 넷으로 학습된 CNN을 가져와서, Object Detection용 데이터 셋으로 fine tuning 한 뒤, selective search 결과로 뽑힌 이미지들로부터 특징 벡터를 추출합니다.
## 3. Classification
> - CNN에서 추출된 특징벡터를 가지고 각각 클래스 별로 SVM에 학습시킨다.
> - 해당 물체가 맞는지 아닌지를 구분하는 classifier모델을 학습시키는것.
> - 왜 SVM사용?
> 그냥 CNN classifier을 쓰는것이 svm을 썼을때보다 성능이 4%낮아졌다. 이는 fine tuning과정에서 물체의 위치정보가 유실되고, 무작위로 추출된 샘플을 학습하여 발생한 것으로 보인다. 
> - 그러나 SVM을 붙여서 학습시키는 기법 역시 더이상 사용하지 않는다.
## 4. Non-Maximum Suppression
> SVM을 통과해 각각의 박스들은 어떤 물체일 확률 값 SCORE을 가지게된다. \
> 그런데 2000개의 박스 모두가 필요할 것인가? 
> - 동일한 물체에 여러 박스가 쳐져있다면 가장 스코어가 높은 박스만 남기고 나머지를 제거한다.\
> ![image](https://user-images.githubusercontent.com/70633080/102711339-40209100-42fc-11eb-9ec6-4474b6801c81.png)
> - 서로다른 두 박스가 동일한 물체인지 어떻게 판별?
> iou(Intersection over union)개념이 적용된다.
> ### IOU
> > - 두 박스의 교집합을 합집합으로 나눠준다.
> > - 두 박스가 일치할 수록 1에 가까운 값이 나온다.
> > ![image](https://user-images.githubusercontent.com/70633080/102711379-968dcf80-42fc-11eb-9b12-3d74ef429668.png)
> > - 논문에서는 iou가 0.5보다 크면 동일한 물체를 대상으로 한 박스라고 판단. 
## 5. Bounding Box Regression
> 앞서에서 selective search를 통해 찾은 박스의 위치는 부정확하다는 문제가 있다.\
> 따라서 성능을 높이기 위해 박스의 위치를 교정해주는 과정을 거친다.\
> 하나의 박스는 다음과 같이 표기될 수 있다.\
> x와 y는 이미지의 중심점 , w와 h는 각각 너비와 높이이다.
> - p_i=(p_ix,p_iy,p_iw,p_ih)\
> ground truth에 해당하는 박스도 다음과 같이 표기될 수 있다.\
> - g=(g_x,g_y,g_w,g_h)
> - 우리의 목표는 P에 해당하는 박스를 최대한 G에 가깝도록 이동시키는 함수를 학습시키는것.\
> 박스가 input으로 들어왔을때 x,y,w,h를 각각 이동시켜주는 함수들을 다음과 같이 표시할 수 있다.
> - dx(p),dy(p),dw(p),dh(p)
> 점 x와 y는 이미지의 크기에 상관없이 위치만 이동시켜주면 된다.\
> w와 h는 이미지의 크기에 비례하여 조정을 시켜주어야 한다.
> - P를 이동시키는 함수의 식
> ![image](https://user-images.githubusercontent.com/70633080/102711871-28e3a280-4300-11eb-96de-8455e0faef65.png)
> - 우리가 학습을 통해 얻고자 하는 것은 d함수이다.\
>  d 함수를 구하기 위해서 앞서 CNN을 통과할 때 pool5 레이어에서 얻어낸 특징 벡터를 사용한다. 그리고 함수에 학습가능한 웨이트 벡터를 주어 계산한다.
> - 이를 식으로 나타내면 아래와 같다.\
>![image](https://user-images.githubusercontent.com/70633080/102748908-6c90e780-43a6-11eb-9dce-253debb7a07f.png)
> - 아래는 loss function이다.
> ![image](https://user-images.githubusercontent.com/70633080/102748999-94804b00-43a6-11eb-9134-fc0a6e654081.png)\
> 일반적인 MSE error function에 L2 normalization을 추가한 형태이며 람다를 1000으로 설정하였다.
> - t는 P를 G로 이동시키기 위해 필요한 이동량을 의미한다.
> ![image](https://user-images.githubusercontent.com/70633080/102749096-bb3e8180-43a6-11eb-90d7-f41d95e95498.png)
> 따라서, CNN을 통과해 추출된 벡터와 x,y,w,h를 조정하는 함수의 weight를 곱해서 bounding box를 조정해주는 선형회귀를 학습시키는 것이다.
