import torch
from torch import nn
from anchor_detect import YoloDetection

def DBL(in_c, out_c, kernel_size, stride, padding):
    dbl_block = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, stride=stride),
                              nn.BatchNorm2d(out_c),
                              nn.LeakyReLU())
    return dbl_block


class Res_unit(nn.Module):
    def __init__(self, in_c):
        super(Res_unit, self).__init__()

        reduce_c = int(in_c / 2)
        self.layer1 = DBL(in_c, reduce_c, 1, 1, 0)
        self.layer2 = DBL(reduce_c, in_c, 3, 1, 1)

    def forward(self, x):
        res_connection = x
        out = self.layer1(x)
        out = self.layer2(out)
        out = out + res_connection
        return out


class Darknet53(nn.Module):
    def __init__(self, block):
        super(Darknet53, self).__init__()

        self.conv1 = DBL(3, 32, 3, 1, 1)
        self.conv2 = DBL(32, 64, 3, 2, 1)

        self.res_block1 = self.num_block(block, 64, num=1)
        self.conv3 = DBL(64, 128, 3, 2, 1)

        self.res_block2 = self.num_block(block, 128, 2)
        self.conv4 = DBL(128, 256, 3, 2, 1)

        self.res_block3 = self.num_block(block, 256, 8)
        self.conv5 = DBL(256, 512, 3, 2, 1)  # 3*3 conv하면 마진1 > 패딩으로 채워줌

        self.res_block4 = self.num_block(block, 512, 8)
        self.conv6 = DBL(512, 1024, 3, 2, 1)

        self.res_block5 = self.num_block(block, 1024, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res_block1(x)
        x = self.conv3(x)
        x = self.res_block2(x)
        x = self.conv4(x)
        x = self.res_block3(x)
        feature3 = x
        x = self.conv5(x)
        x = self.res_block4(x)
        feature2 = x
        x = self.conv6(x)
        x = self.res_block5(x)
        feature1 = x

        return feature1, feature2, feature3

    def num_block(self, block, in_c, num):
        layers = []
        for i in range(num):
            layers.append(Res_unit(in_c))
        return nn.Sequential(*layers)


class Yolo_v3(nn.Module):
    def __init__(self):
        super(Yolo_v3, self).__init__()
        self.class_num = 80
        self.img_size = 416
        anchor = {'grid52': [(10, 13), (16, 30), (33, 23)],
                  'grid26': [(30, 61), (62, 45), (59, 119)],
                  'grid13': [(116, 90), (156, 198), (373, 326)]}

        self.darknet53 = Darknet53(Res_unit)

        self.conv_set1 = self.conv_set(1024, 512)
        self.conv_final1 = self.conv_final(512, 255)
        self.anchor_box1 = YoloDetection(anchor['grid13'], self.img_size, self.class_num)

        self.conv_layer1 = DBL(512, 256, 1, 1, 0)
        self.upsampling1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_set2 = self.conv_set(768, 256)  # 왜 나누기 3???????????
        self.conv_final2 = self.conv_final(256, 255)
        self.anchor_box2 = YoloDetection(anchor['grid26'], self.img_size, self.class_num)

        self.conv_layer2 = DBL(256, 128, 1, 1, 0)
        self.upsampling2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_set3 = self.conv_set(384, 128)
        self.conv_final3 = self.conv_final(128, 255)
        self.anchor_box3 = YoloDetection(anchor['grid52'], self.img_size, self.class_num)

    def forward(self, x, target=None):
        targets=None

        print("darknet53 ")
        res5, res4, res3 = self.darknet53(x)

        print("=========================================")
        print("yolov3")
        #         print("res3: ",res3.shape) # res 1이후
        #         print("res4: ",res4.shape) # res 3이후
        #         print("res5: ",res5.shape)# 마지막

        # 1번째 feature 뽑기
        out1 = self.conv_set1(res5)
        first = self.conv_final1(out1)

        anchor13, loss_layer1 = self.anchor_box1(first,targets)  # [1,507,85]

        # 2번째 feature
        out2 = self.conv_layer1(out1)
        out2 = self.upsampling1(out2)
        out2 = torch.cat((out2, res4), dim=1)
        #         print("concate1_result:",out2.shape)
        out2 = self.conv_set2(out2)
        second = self.conv_final2(out2)

        anchor26, loss_layer2 = self.anchor_box2(second,targets)  # [1, 2028, 85]

        # 3번째 feature
        out3 = self.conv_layer2(out2)
        out3 = self.upsampling2(out3)
        out3 = torch.cat((out3, res3), dim=1)
        #         print("concate2_result:", out3.shape)
        out3 = self.conv_set3(out3)
        thrid = self.conv_final3(out3)

        anchor52, loss_layer3 = self.anchor_box3(thrid,targets)  # [1, 8112, 85]

        # feature 크기출력
        print(">>>>> featuremap extract <<<<<")
        print("first_feature:", first.shape)
        print("second_feature:", second.shape)
        print("thrid_feature:", thrid.shape)

        # anchor box합치기
        print(">>>>> anchor box prediction <<<<<")
        yolo_output = [anchor13, anchor26, anchor52]
        yolo_output = torch.cat(yolo_output, 1).detach()  # 인덱스1번째 차원으로 합치기. shape: [1,10647,85]

        # 최종 loss
        loss = loss_layer1 + loss_layer2 + loss_layer3

        return yolo_output if target is None else (loss, yolo_output)

    def conv_set(self, in_c, out_c):
        increase_c = out_c * 2
        result = nn.Sequential(DBL(in_c, out_c, 1, 1, 0),
                               DBL(out_c, increase_c, 3, 1, 1),
                               DBL(increase_c, out_c, 1, 1, 0),
                               DBL(out_c, increase_c, 3, 1, 1),
                               DBL(increase_c, out_c, 1, 1, 0))
        return result

    def conv_final(self, in_c, out_c):
        result = nn.Sequential(DBL(in_c, in_c * 2, 3, 1, 1),
                               nn.Conv2d(in_c * 2, out_c, 1, 1, 0))
        return result


# 입력이미지 랜덤생성
input_image=torch.randn(1,3,416,416)
print(input_image.shape)

Yolo_v3().forward(x=input_image)