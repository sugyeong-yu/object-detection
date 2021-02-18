import numpy as np
import torch
from torch import nn
from models.anchor_detect import YoloDetection


def DBL(in_c, out_c, kernel_size, stride, padding):
    module1 = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, stride=stride)
    module2 = nn.BatchNorm2d(out_c)
    dbl_block = nn.Sequential(module1,
                              module2,
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

        self.conv_1 = DBL(3, 32, 3, 1, 1)
        self.conv_2 = DBL(32, 64, 3, 2, 1)

        self.res_block1 = self.num_block(block, 64, num=1)
        self.conv_3 = DBL(64, 128, 3, 2, 1)

        self.res_block2 = self.num_block(block, 128, 2)
        self.conv_4 = DBL(128, 256, 3, 2, 1)

        self.res_block3 = self.num_block(block, 256, 8)
        self.conv_5 = DBL(256, 512, 3, 2, 1)  # 3*3 conv하면 마진1 > 패딩으로 채워줌

        self.res_block4 = self.num_block(block, 512, 8)
        self.conv_6 = DBL(512, 1024, 3, 2, 1)

        self.res_block5 = self.num_block(block, 1024, 4)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.res_block1(x)
        x = self.conv_3(x)
        x = self.res_block2(x)
        x = self.conv_4(x)
        x = self.res_block3(x)
        feature3 = x
        x = self.conv_5(x)
        x = self.res_block4(x)
        feature2 = x
        x = self.conv_6(x)
        x = self.res_block5(x)
        feature1 = x

        return feature1, feature2, feature3

    def num_block(self, block, in_c, num):
        layers = []
        for i in range(num):
            layers.append(Res_unit(in_c))
        return nn.Sequential(*layers)


class Yolo_v3(nn.Module):
    def __init__(self, image_size: int, class_num: int):
        super(Yolo_v3, self).__init__()
        self.class_num = class_num
        self.img_size = image_size
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

        self.yolo_layers=[self.anchor_box3,self.anchor_box2,self.anchor_box1]

    def forward(self, x, target):
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

        anchor13, loss_layer1 = self.anchor_box1(first, target)  # [1,507,85]

        # 2번째 feature
        out2 = self.conv_layer1(out1)
        out2 = self.upsampling1(out2)
        out2 = torch.cat((out2, res4), dim=1)
        #         print("concate1_result:",out2.shape)
        out2 = self.conv_set2(out2)
        second = self.conv_final2(out2)

        anchor26, loss_layer2 = self.anchor_box2(second, target)  # [1, 2028, 85]

        # 3번째 feature
        out3 = self.conv_layer2(out2)
        out3 = self.upsampling2(out3)
        out3 = torch.cat((out3, res3), dim=1)
        #         print("concate2_result:", out3.shape)
        out3 = self.conv_set3(out3)
        thrid = self.conv_final3(out3)

        anchor52, loss_layer3 = self.anchor_box3(thrid, target)  # [1, 8112, 85]

        # # feature 크기출력
        # print(">>>>> featuremap extract <<<<<")
        # print("first_feature:", first.shape)
        # print("second_feature:", second.shape)
        # print("thrid_feature:", thrid.shape)

        # anchor box합치기
        #print(">>>>> anchor box prediction <<<<<")
        yolo_output = [anchor13, anchor26, anchor52]
        yolo_output = torch.cat(yolo_output, 1).detach()  # 인덱스1번째 차원으로 합치기. shape: [1,10647,85] # 10647= 모든 앵커수 * grid

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

    def load_darknet_weights(self, weights_path: str):
        # Open the weights file
        with open(weights_path, "rb") as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values (0~2: version, 3~4: seen)
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        ptr = 0
        # Load Darknet-53 weights
        for name, modules in self.darknet53.named_modules():
            # print("name:", name)
            module_type = name.split('_')[0]
            if module_type == 'conv':
                if str(type(modules)) == "<class 'torch.nn.modules.container.Sequential'>":
                    #print(modules[0])
                    ptr = self.load_bn_weights(modules[1], weights, ptr)  # module[1]은 DBL에서의 batch
                    ptr = self.load_conv_weights(modules[0], weights, ptr)  # [0]은 DBL에서의 conv

            if module_type == 'res':
                if str(type(modules)) == "<class 'torch.nn.modules.container.Sequential'>":
                    if len(modules) == 3:
                        # 이거 안되면 레스넷 시퀀셜로 묶고 [i][0] 식으로 진행.
                        ptr = self.load_bn_weights(modules[1], weights, ptr)
                        ptr = self.load_conv_weights(modules[0], weights, ptr)
        # Load YOLOv3 weights
        # conv_set,conv_final,upsampling에 대해서만 전이학습
        if weights_path.find('yolov3.weights') != -1:
            # conv_set은 DBL5번한ㄴ conv_set 함수
            # conv_final은 DBL , conv 한번씩
            for module in self.conv_set1:
                # module에는 conv_set의 DBL5개 중 하나씩 순차적으로.
                ptr = self.load_bn_weights(module[1], weights, ptr) # module[1]은 DBL에서의 batch
                ptr = self.load_conv_weights(module[0], weights, ptr) # [0]은 DBL에서의 conv

            ptr = self.load_bn_weights(self.conv_final1[0][1], weights, ptr) # conv_final[0]은 DBL, [0][1]은 DBL에서의 batch
            ptr = self.load_conv_weights(self.conv_final1[0][0], weights, ptr) #[0][1]은 DBL에서의 conv
            ptr = self.load_conv_bias(self.conv_final1[1], weights, ptr) # conv_final[1]은 conv layer
            ptr = self.load_conv_weights(self.conv_final1[1], weights, ptr)

            ptr = self.load_bn_weights(self.upsampling1[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.upsampling1[0][0], weights, ptr)

            for module in self.conv_set2:
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            ptr = self.load_bn_weights(self.conv_final2[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final2[0][0], weights, ptr)
            ptr = self.load_conv_bias(self.conv_final2[1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final2[1], weights, ptr)

            ptr = self.load_bn_weights(self.upsampling2[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.upsampling2[0][0], weights, ptr)

            for module in self.conv_set3:
                ptr = self.load_bn_weights(module[1], weights, ptr)
                ptr = self.load_conv_weights(module[0], weights, ptr)

            ptr = self.load_bn_weights(self.conv_final3[0][1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final3[0][0], weights, ptr)
            ptr = self.load_conv_bias(self.conv_final3[1], weights, ptr)
            ptr = self.load_conv_weights(self.conv_final3[1], weights, ptr)

    # Load BN bias, weights, running mean and running variance
    def load_bn_weights(self, bn_layer, weights, ptr: int):
        num_bn_biases = bn_layer.bias.numel()  # bias.numel > layer의 weight 수를 얻는다.

        # Bias
        # ptr : 시작점, ptr + num_bn_biases : 종료점 (시작+불러올 weight수)
        bn_biases = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.bias)  # 불러올 새 편향
        bn_layer.bias.data.copy_(bn_biases)  # 복사.
        ptr += num_bn_biases
        # Weight
        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.weight)
        bn_layer.weight.data.copy_(bn_weights)
        ptr += num_bn_biases
        # Running Mean
        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.running_mean)
        bn_layer.running_mean.data.copy_(bn_running_mean)
        ptr += num_bn_biases
        # Running Var
        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases]).view_as(bn_layer.running_var)
        bn_layer.running_var.data.copy_(bn_running_var)
        ptr += num_bn_biases

        # weight를 몇번째 index까지 사용했는지를 나타내는 정수 ptr 을 반환
        return ptr

    # Load convolution bias
    def load_conv_bias(self, conv_layer, weights, ptr: int):
        num_biases = conv_layer.bias.numel()

        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases]).view_as(conv_layer.bias) # convlayer의 편향
        conv_layer.bias.data.copy_(conv_biases)
        ptr += num_biases

        return ptr

    # Load convolution weights
    def load_conv_weights(self, conv_layer, weights, ptr: int):
        num_weights = conv_layer.weight.numel()

        conv_weights = torch.from_numpy(weights[ptr: ptr + num_weights])
        conv_weights = conv_weights.view_as(conv_layer.weight)
        conv_layer.weight.data.copy_(conv_weights)
        ptr += num_weights

        return ptr

# 입력이미지 랜덤생성
# input_image=torch.randn(1,3,416,416)
# print(input_image.shape)
#
# Yolo_v3().forward(x=input_image)
# model=Darknet53(Res_unit)
# for name, module in model.named_modules():
#     print("name:",name)
#     module_type = name.split('_')[0]
#     if module_type == 'conv':
#         if str(type(module)) == "<class 'torch.nn.modules.container.Sequential'>":
#               print(module[0])
#
#     if module_type == 'res':
#         if str(type(module)) == "<class 'torch.nn.modules.container.Sequential'>":
#             if len(module) == 3 :
#                 print(module)
#
# model=Yolo_v3(416,80)
# for module in model.conv_set1:
#     print("module:", module[0])