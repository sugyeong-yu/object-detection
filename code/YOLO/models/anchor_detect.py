import torch
from torch import nn
from utils.utils import matching_target


class YoloDetection(nn.Module):
    def __init__(self, anchor, img_size, classnum):
        super(YoloDetection, self).__init__()
        self.anchor = anchor
        self.img_size = img_size
        self.numclass = classnum
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ignore_thres = 0.5

        self.obj_weight = 1
        self.no_obj_weight = 100

    def forward(self, x , targets):
        batch_size = x.size(0)
        num_anchor = len(self.anchor)
        grid_size = x.size(2)  # 13-> 26-> 52

        pred = (x.view(batch_size, num_anchor, 5 + self.numclass, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2).contiguous())
        print(pred.shape)  # batch, anchor,img,img,5+class

        bx = torch.sigmoid(pred[..., 0])  # 시작은 그냥 특징맵의 x좌표로 변화량 계산 -> 학습을 통해 변화
        by = torch.sigmoid(pred[..., 1])  # 모두 0~1 사이값 즉, 변화량
        w = pred[..., 2]
        h = pred[..., 3]
        pred_conf = torch.sigmoid(pred[..., 4])  # object confidence
        pred_cls = torch.sigmoid(pred[..., 5:])  # class prediction

        stride = self.img_size / grid_size  # 416/13 = 32
        left_x = torch.arange(grid_size, dtype=torch.float).repeat(grid_size, 1).view([1, 1, grid_size, grid_size])
        left_y = torch.arange(grid_size, dtype=torch.float).repeat(grid_size, 1).t().view(
            [1, 1, grid_size, grid_size])  # y니까 전치해주기 t()

        # grid 나눈거에 맞춰서 anchor크기도 맞춰주기 ex) 4 -> 2
        grid_anchor = torch.as_tensor([(anchor_w / stride, anchor_h / stride) for anchor_w, anchor_h in self.anchor],
                                      dtype=float)

        anchor_w = grid_anchor[:, 0].view((1, num_anchor, 1, 1))
        anchor_h = grid_anchor[:, 1].view((1, num_anchor, 1, 1))  # 왜이런 형태로?

        # 상대좌표 구하기 (grid상 좌표) 좌상단 + 변화량
        pred_bbox = torch.zeros_like(pred[..., :4])  # x,y,w,h 크기의 예측박스
        pred_bbox[..., 0] = left_x + bx  # 좌상단좌표 + 변화량
        pred_bbox[..., 1] = left_y + by
        pred_bbox[..., 2] = torch.exp(w) * anchor_w
        pred_bbox[..., 3] = torch.exp(h) * anchor_h

        # print(pred_bbox.shape) # (1,3,13,13,4)

        # x,y,w,h 와 conf, cls 합쳐주기
        # 절대좌표구하기 (실제이미지상 좌표)
        pred = (pred_bbox.view(batch_size, -1, 4) * stride,  # (1,507,4)
                pred_conf.view(batch_size, -1, 1),  # (1,507,1)
                pred_cls.view(batch_size, -1, self.numclass))  # (1,507,80)
        output = torch.cat(pred, -1)  # 마지막차원을 기준으로 합친다. (1,507,85)

        #         if target is None :
        #             print("target is none")
        #             return output, 0

        iou_score, class_mask,obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf = matching_target(pred_bbox,targets,pred_cls,self.anchor,self.ignore_thres)

        #loss 계산 예측변화량 - 실제변화량
        loss_x = self.mse_loss(bx[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(by[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask],tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], ty[obj_mask])
        loss_bbox = loss_x + loss_y + loss_w + loss_h

        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_no_obj = self.bce_loss(pred_conf[no_obj_mask], tconf[no_obj_mask])
        loss_conf = (self.obj_weight * loss_conf_obj) + (self.no_obj_weight * loss_conf_no_obj) # scale은 뭘까

        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

        loss_layer = loss_bbox + loss_conf + loss_cls

        # loss와 최종 절대좌표의 pred box를 출력한다.
        return output, loss_layer