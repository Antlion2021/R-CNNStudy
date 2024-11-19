import torch
import torch.nn as nn
import torchvision
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RPN(nn.Module):
    def __init__(self, in_channels=512):
        super(RPN, self).__init__()
        self.scales = [128, 256, 512]
        self.aspect_ratios = [0.5, 1.0, 2.0]
        # 3*3 conv
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)

        self.rpn_conv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        # 1*1 conv:classification
        self.cls_layer = nn.Conv2d(in_channels,
                                   self.num_anchors,
                                   kernel_size=1,
                                   stride=1)
        # 1*1 regression
        self.bbox_reg_layer = nn.Conv2d(in_channels,
                                        self.num_anchors * 4,
                                        kernel_size=1,
                                        stride=1)
