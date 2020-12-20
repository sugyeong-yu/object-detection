# R-CNN
![image](https://user-images.githubusercontent.com/70633080/102708762-3e4cd280-42e8-11eb-82ae-273588515824.png)
- R-CNN의 Object detection 알고리즘
1. 입력이미지에 selective search 알고리즘을 적용해 물체가 있을만한 박스 2000개 추출
2. 모든 박스를 227*227 사이즈로 resize. (박스의 비율은 고려하지 않는다.)
3. 이미지넷 데이터를 통해 학습된 CNN을 통과시켜 4096차원의 특징벡터를 추출.
4. 추출된 벡터로 각 클래스마다 학습시켜놓은 SVM Classifier를 통과.
5. 바운딩 박스 리그레션을 적용해 박스의 위치를 조정

## 1. Region Proposal
> Region Proposal이란 주어진 이미지에서 물체가 있을법한 위치를 찾는것이다.\
> R-cnn은 " Selective Search "라는 룰베이스 알고리즘을 통해 2000개의 물체박스를 찾는다.\
> ### Selective Search
> > 주변 픽셀 간의 유사도를 기준으로 Segmentation을 만들고, 이를 기준으로 물체가 있을법한 박스를 추론\
> > ![image](https://user-images.githubusercontent.com/70633080/102708835-c3d08280-42e8-11eb-872e-e4af63ccfb51.png)
> - 그러나 R-CNN이후 region proposal 과정은 뉴럴 네트워크가 수행하도록 발전되었다. 따라 더이상 사용하지 않는 알고리즘이다.
## 2. Feature Extraction
> Selective Search를 통해서 찾아낸 2천개의 박스 영역은 227 x 227 크기로 리사이즈 됩니다.\
> 그리고 Image Classification으로 미리 학습되어 있는 CNN 모델을 통과하여 4096 크기의 특징 벡터를 추출한다.\
> - 미리학습된 모델이란?
> 
