import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class DepthSeperabelConv2d(nn.Module)        :
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super(DepthSeperabelConv2d, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size, groups=input_channels, **kwargs),
            nn.BatchNorm2d(input_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, input_channels, output_cahnnels, kernel_size, padding=0, stride=1, bias=True):
        super(BasicConv2d, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, output_cahnnels, kernel_size, padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(output_cahnnels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class MobileNet_V1(nn.Module):
    # alpha (0, 1], 一般为 1, 1/2, 1/4, ...
    def __init__(self, alpha=1, init_weights=True):
        super().__init__()

        # no downsample
        self.layer0 = nn.Sequential(
            BasicConv2d(3, int(32*alpha), 3, padding=1, bias=False),
            DepthSeperabelConv2d(int(32*alpha), int(64*alpha), 3, padding=1, bias=False)
        )
        self.layer1 = nn.Sequential(
            DepthSeperabelConv2d(int(64*alpha), int(128*alpha), 3, stride=2, padding=1, bias=False),#downsample
            DepthSeperabelConv2d(int(128*alpha), int(128*alpha), 3, padding=1, bias=False)
        )
        # /4
        self.layer2 = nn.Sequential(
            DepthSeperabelConv2d(int(128*alpha), int(256*alpha), 3, stride=2, padding=1, bias=False),#downsample
            DepthSeperabelConv2d(int(256*alpha), int(256*alpha), 3, padding=1, bias=False)
        )

        # /8
        self.layer3 = nn.Sequential(
            DepthSeperabelConv2d(int(256*alpha), int(512*alpha), 3, stride=2, padding=1, bias=False),#downsample
            DepthSeperabelConv2d(int(512*alpha), int(512*alpha), 3, padding=1, bias=False),
            DepthSeperabelConv2d(int(512*alpha), int(512*alpha), 3, padding=1, bias=False),
            DepthSeperabelConv2d(int(512*alpha), int(512*alpha), 3, padding=1, bias=False),
            DepthSeperabelConv2d(int(512*alpha), int(512*alpha), 3, padding=1, bias=False),
            DepthSeperabelConv2d(int(512*alpha), int(512*alpha), 3, padding=1, bias=False)
        )

        # /16
        self.layer4 = nn.Sequential(
            DepthSeperabelConv2d(int(512*alpha), int(1024*alpha), 3, stride=2, padding=1, bias=False),#downsample
            DepthSeperabelConv2d(int(1024*alpha), int(1024*alpha), 3, padding=1, bias=False)
        )
        
        self.out_channels_list = [int(256*alpha), int(512*alpha), int(1024*alpha)]

        if init_weights:
            self.__init_weights()
    
    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        return x1, x2, x3


class FPN(nn.Module):
    def __init__(self, channels_list, out_channels):
        super().__init__()
        self.out1 = BasicConv2d(channels_list[0], out_channels, kernel_size=1)
        self.out2 = BasicConv2d(channels_list[1], out_channels, kernel_size=1)
        self.out3 = BasicConv2d(channels_list[2], out_channels, kernel_size=1)
        
        self.conv_trans_3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv_trans_2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        
        self.merge1 = BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.merge2 = BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
    
    def forward(self, x):
        x1, x2, x3 = x
        
        x1 = self.out1(x1)
        x2 = self.out2(x2)
        out3 = self.out3(x3)
        
        # up3 = F.interpolate(out3, size=x2.shape[2:4], mode="nearest")
        up3 = self.conv_trans_3(out3)
        x2 = x2 + up3
        out2 = self.merge2(x2)
        
        # up2 = F.interpolate(out2, size=x1.shape[2:4], mode="nearest")
        up2 = self.conv_trans_2(out2)
        x1 = x1 + up2
        out1 = self.merge1(x1)
        return out1, out2, out3
    

class conv_bn_no_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride),
            nn.BatchNorm2d(out_channel),
        )
    def forward(self, x):
        return self.layers(x)
    
class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        assert out_channel % 4 == 0
        
        self.conv3x3 = conv_bn_no_relu(in_channel, out_channel//2, kernel_size=3, stride=1)
        
        self.conv5x5_1 = BasicConv2d(in_channel, out_channel//4, kernel_size=3, padding=1, stride=1)
        self.conv5x5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, kernel_size=3, stride=1)
        
        self.conv7x7_2 = BasicConv2d(out_channel//4, out_channel//4, kernel_size=3, padding=1, stride=1)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, kernel_size=3, stride=1)
    
    def forward(self, x):
        conv3x3 = self.conv3x3(x)
        
        conv5x5_1 = self.conv5x5_1(x)
        conv5x5_2 = self.conv5x5_2(conv5x5_1)
        
        conv7x7 = self.conv7x7_3(self.conv7x7_2(conv5x5_1))
        
        out = torch.cat([conv3x3, conv5x5_2, conv7x7], dim=1)
        return F.relu(out)
    

class Head(nn.Module):
    def __init__(self,in_channels, out_channels, d):
        super().__init__()
        self.d = d
        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    def forward(self, x):
        out = self.layer(x)
        b, _, h, w = out.shape
        return out.permute(0, 2, 3, 1).contiguous().reshape(b, h, w, -1, self.d)
        
# config:
#   “backbone_scale”
#   "out_channel"
#   "ratios"
#   "num_landmark_points"
class RetinaFace(nn.Module):
    def __init__(self, config, num_classes=1, mode="train"):
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.body = MobileNet_V1(config["backbone_scale"])
        
        self.fpn = FPN(self.body.out_channels_list, config["out_channel"])
        
        self.ssh1 = SSH(config["out_channel"], config["out_channel"])
        self.ssh2 = SSH(config["out_channel"], config["out_channel"])
        self.ssh3 = SSH(config["out_channel"], config["out_channel"])
        
        self.ClsHeads = nn.ModuleList([Head(config["out_channel"], 2*len(r)*(num_classes+1), num_classes+1) for r in config["ratios"]])
        self.BboxHeads = nn.ModuleList([Head(config["out_channel"], 2*len(r)*4, 4) for r in config["ratios"]])
        self.LandmarkHeads = nn.ModuleList([Head(config["out_channel"], 2*len(r)*config["num_landmark_points"]*2, config["num_landmark_points"]*2) for r in config["ratios"]])
        
    def forward(self, x):
        out = self.body(x)
        fpn1, fpn2, fpn3 = self.fpn(out)
        
        features = [self.ssh1(fpn1), self.ssh2(fpn2), self.ssh3(fpn3)]
        # print([i.shape for i in features])
        
        # F.cross_entropy 为 Softmax–Log–NLLLoss，故不需softmax，但是infer的时候还是需要的
        if self.mode == "train":
            classifications = [head(feature) for head, feature in zip(self.ClsHeads, features)]
        else:
            classifications = [torch.softmax(head(feature), dim=-1) for head, feature in zip(self.ClsHeads, features)]
        
        bbox_regression = [head(feature) for head, feature in zip(self.BboxHeads, features)]
        landmark_regression = [head(feature) for head, feature in zip(self.LandmarkHeads, features)]
        return classifications, bbox_regression, landmark_regression
        

    