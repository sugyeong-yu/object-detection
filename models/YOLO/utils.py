import torch
from torch import nn


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
    obj_mask[batch_size,best_iou_idxs,gj,gi] = 1 #물체가 있는 곳을 1로 만들어줌 ex) 0번쨰 물체의 anchor idx는 0이고 실제 중심좌표는 1,1일때 0의 1,1을 True로.
    no_obj_mask[batch_size, best_iou_idxs, gj, gi] = 0 # 물체가 있는 곳을 0으로 만들어줌

    # iou가 임계값보다 클 경우, no_obj_mask 에서 0으로 만들기 (물체가 있다고 판단)
    for i, anchor_ious in enumerate(ious.t()):
        no_obj_mask[batch[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # ground truth 좌표의 변화량 구하기 (offset)
    tx[batch,best_iou_idxs,gj,gi]=gx-gx.floor() # 해당 anchor box의 실제 물체 좌표에만 값이 들어가게됨.
    ty[batch,best_iou_idxs, gj, gi] = gy - gy.floor()
    tw[batch,best_iou_idxs,gj,gi] = torch.log(gw/anchors[best_iou_idxs][:,0]+1e-16) # 여기 왜 이런 연산?
    th[batch,best_iou_idxs,gj,gi] = torch.log(gh/anchors[best_iou_idxs][:,1]+1e-16)

    #one-hot encoding of label
    tcls[batch,best_iou_idxs,gj,gi,target_labels] = 1 # 물체마다 해당하는 anchor인덱스에서 실제물체의 좌표에 1을 넣어줌.

    # 물체에 해당하는 ioubox 인덱스의 feature map에서 물체의 실제좌표의 클래스 = 예측한것과 target이 맞는경우 1, 틀리면 0
    class_mask[batch_size, best_iou_idxs, gj, gi] = (pred_cls[batch_size, best_iou_idxs, gj, gi].argmax(-1) == target_labels).float()
    iou_score[batch_size, best_iou_idxs, gj, gi] = bbox_iou(pred_box[batch_size, best_iou_idxs, gj, gi], target_box, x1y1x2y2=False) # bbox iou계싼

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
