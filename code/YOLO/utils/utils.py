import torch, torchvision
from torch import nn


def corner(x):
    # x는 [x,y,w,h] 형태
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def NMS(pred_box,conf_thres,nms_thres):
    # pred_box>[1,10647,85]
    # 1. conf thres보다 작은 bbox삭제
    # 2. class score 순으로 정렬
    # 3. NMS수행 : class 별로 다음 박스와 비교하고 iou가 일정 thres를 넘으면 동일 객체를 detect한것으로 판별하고 0으로 만들어줌.
    # return > (x1, y1, x2, y2, object_conf, class_score, class_pred)
    pred_box[...,:4] = corner(pred_box[...,:4]) # 점들을 코너로 바꿔줌 #[1, 10647, 85]
    output = [None for _ in range(len(pred_box))] # out shape정의

    # batch마다 가져옴.  image가 1장일때, batch가 1이므로 반복문이 1번돈다.
    for img_i,img_pred in enumerate(pred_box):
        print(img_pred.shape) #[10647, 85]
        # 1. confidence score가 thres 넘는거만 통과
        img_pred = img_pred[img_pred[:,4] >= conf_thres] # True/False로 저장 #[?,85]
        if not img_pred.size(0):
            continue

        # score계산 (conf * class)
        score = img_pred[:,4] * img_pred[:,5:].max(1)[0] # 클래스 점수 제일 큰 클래스값와 conf를 곱한다.

        # 정렬 ( 큰순으로 정렬하기 위해서 score에 -를 붙임)
        img_pred = img_pred[(-score).argsort()] # argsort(dim=1) > 행마다 각 열에서 값이 낮은 순으로 인덱스로 저장., img_pred는 score큰 순으로 정렬
        class_confs, class_preds = img_pred[:, 5:].max(1, keepdim=True)#(값, 인덱스) keepdim=True > 해당차원을 제외하고 출력tensor는 동일한 크기의 dim을 유지, 가장 큰 class score과 인덱스를 출력
        detections = torch.cat((img_pred[:, :5], class_confs.float(), class_preds.float()), 1) # 열을 기준으로 합침. 열이 늘어남. #[?,7]


        keep_box = []
        # anchorbox가 있는 만큼 반복문 실행
        while detections.size(0):
            # score가 가장높은 첫번째 박스와 가장 iou가 큰 anchor box를 찾는다.
            large_iou = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0,-1] == detections[:,-1] # score가장큰 anchor box와 class label이 같은가 True/False

            # iou가 thres를 넘고 class label이 동일한 anchor box의 confidence score 를 weights라는 변수에 저장
            invalid = large_iou & label_match # 둘다 참일때만 True ( 조건에 맞는 box들만 받아오기 위함)
            weights = detections[invalid, 4:5] # invalid가 True인 행? 의 confidence

            # [0,:4] > 0번index의 batch에서 index3번까지의 anchorbox를 선택 >>따라서 처음 배치shape만 사라지고 shape은 같음
            # confidence score 별로 좌표값을 곱해 더함으로써 score가 가장큰 box의 좌표값을 조정하여 최종 좌표값을 구함.
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum() # sum(0) > 각 행마다 요소별로 더함(x는 x끼리 y는 y끼리...)
            keep_box += [detections[0]]
            detections = detections[~invalid] # 이제 나머지 중에서, 즉 False였던 것 중에서 시작.
        if keep_box:
            output[img_i] = torch.stack(keep_box) # list -> torch로 변경

        return output

test = torch.zeros(1,10647,85)
NMS(test,1,1)

def bbox_wh_iou(wh1, wh2):
    print("wh1:",wh1)
    print("wh2:", wh2)
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

def bbox_iou(box1,box2,x1y1x2y2=True):
    # 2개의 box에 대한 iou를 계산하고 return해준다.
    # x1y1x2y2가 False면 좌상단 우하단으로 바꿔줌잠ㅁ (계산의 편리성 위해)
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else :
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # inter
    # 교차되는 직사각형의 좌표를 얻는다.
    inter_x1 = torch.max(b1_x1,b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    # clamp는 min혹은 max의 범주에 해당되도록 값을 변경하는 것을 의미한다.
    # ex) 2, 3, 5가 있을때 min=4라고 한다면 최소가 4가 되도록 이하의 값들을 교체한다.
    inter_area = torch.clamp(inter_x2-inter_x1+1,min=0) * torch.clamp(inter_y2-inter_y1+1,min=0) # 왜 +1???

    # union
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

def matching_target(pred_box,target,pred_cls,anchors,ignore_thres):
    # pred_box의 shape : (1,3,13,13,4) >> 13*13 픽셀 하나 = grid1개 당 앵커박스 3개있음
    batch_size = pred_box.size(0) #배치사이즈 , 1
    print("batxch",batch_size)
    num_anchor = pred_box.size(1) #앵커박스개수 ,3
    num_class = pred_cls.size(-1) #class 개수
    grid_size = pred_box.size(2) #grid size 13->26->52

    # make mask
    obj_mask = torch.zeros(batch_size,num_anchor,grid_size,grid_size,dtype=torch.bool) # 물체가 없으면 0, 있으면 1
    no_obj_mask = torch.ones(batch_size,num_anchor,grid_size,grid_size,dtype=torch.bool) # 물체가 없으면 1, 있으면 0
    class_mask = torch.zeros(batch_size,num_anchor,grid_size,grid_size,dtype=torch.float) # class에 해당하지않으면 0, 해당하면 1
    iou_score = torch.zeros(batch_size,num_anchor,grid_size,grid_size,dtype=torch.float)
    # true좌표를 담을 텐서초기화
    tx = torch.zeros(batch_size,num_anchor,grid_size,grid_size,dtype=torch.float)
    ty = torch.zeros(batch_size,num_anchor,grid_size,grid_size,dtype=torch.float)
    tw = torch.zeros(batch_size,num_anchor,grid_size,grid_size,dtype=torch.float)
    th = torch.zeros(batch_size,num_anchor,grid_size,grid_size,dtype=torch.float)
    tcls = torch.zeros(batch_size,num_anchor,grid_size,grid_size,num_class,dtype=torch.float)

    target_box = target[ : ,2:6] * grid_size # target은 1*1 크기가 기준 따라서 같은 grid로 비교해주기 위해 grid크기만큼 곱함
    gxy = target_box[ : , :2] # ground truth의 x,y
    gwh = target_box[ : , 2:] # gruond truth의 w,h

    # iou 값이 큰 anchor box를 찾는다. (w와 h만 비교)
    iou=torch.stack([bbox_wh_iou(anchor,gwh) for anchor in anchors])
    best_iou_value,best_iou_idxs = iou.max(0) # 열의 개수만큼 max값 추출, 해당 열에서의 각 행중 max (물체수,앵커박스수) >> 물체 마다 3개의 앵커박스 중 어느게 젤 크기가 비슷한가

    # target 분리하기
    batch,target_labels=target[:,:2].long().t() # t()는 transpose
    gx,gy = gxy.t()
    gw,gh = gwh.t()
    gi,gj = gxy.long().t() # 물체의 실제 중심좌표. 따라서 best_iou_idxs의 길이? 개수 와 동일하다.

    # set masks
    obj_mask[batch,best_iou_idxs,gj,gi] = 1 #물체가 있는 곳을 1로 만들어줌 ex) 0번쨰 물체의 anchor idx는 0이고 실제 중심좌표는 1,1일때 0의 1,1을 True로.
    no_obj_mask[batch, best_iou_idxs, gj, gi] = 0 # 물체가 있는 곳을 0으로 만들어줌

    # iou가 임계값보다 클 경우, no_obj_mask 에서 0으로 만들기 (물체가 있다고 판단)
    for i, anchor_ious in enumerate(iou.t()):
        no_obj_mask[batch[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # ground truth 좌표의 변화량 구하기 (offset)
    tx[batch,best_iou_idxs,gj,gi]=gx-gx.floor() # 해당 anchor box의 실제 물체 좌표에만 값이 들어가게됨.
    ty[batch,best_iou_idxs, gj, gi] = gy - gy.floor()
    tw[batch,best_iou_idxs,gj,gi] = torch.log(gw/anchors[best_iou_idxs][:,0]+1e-16) # 여기 왜 이런 연산?
    th[batch,best_iou_idxs,gj,gi] = torch.log(gh/anchors[best_iou_idxs][:,1]+1e-16)

    #one-hot encoding of label
    tcls[batch,best_iou_idxs,gj,gi,target_labels] = 1 # 물체마다 해당하는 anchor인덱스에서 실제물체의 좌표에 1을 넣어줌.

    # 물체에 해당하는 ioubox 인덱스의 feature map에서 물체의 실제좌표의 클래스 = 예측한것과 target이 맞는경우 1, 틀리면 0
    class_mask[batch, best_iou_idxs, gj, gi] = (pred_cls[batch, best_iou_idxs, gj, gi].argmax(-1) == target_labels).float()
    iou_score[batch, best_iou_idxs, gj, gi] = bbox_iou(pred_box[batch, best_iou_idxs, gj, gi], target_box, x1y1x2y2=False) # bbox iou계싼

    tconf = obj_mask.float()
    return iou_score, class_mask, obj_mask, no_obj_mask, tx, ty, tw,th, tcls, tconf







# # best_ious랑 objmask 동작확인용 테스트코드
# a=torch.tensor([(10, 13), (16, 30), (33, 23)])
# gwh =torch.tensor([(10, 13), (16, 30), (33, 23),(50,40)])
#
# ious = torch.stack([bbox_wh_iou(i, gwh) for i in a])
# _,best_iou_idxs=ious.max(0)
# print(ious)
# print(ious.shape)
# print(ious.max(0))
#
# gi=torch.tensor([1,2,3,4])
# gj=torch.tensor([1,2,3,4])
#
# obj_mask = torch.zeros(1, 3, 13, 13, dtype=torch.bool)
# obj_mask[0,best_iou_idxs,gj,gi] = 1
# print(obj_mask)