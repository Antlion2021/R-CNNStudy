import torch
import torch.nn as nn
import torchvision
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RPN(nn.Module):  # R-CNN RPN part: First Layer
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
        # 1*1 Classification
        self.cls_layer = nn.Conv2d(in_channels,
                                   self.num_anchors,
                                   kernel_size=1,
                                   stride=1)
        # 1*1 Box regression
        self.bbox_reg_layer = nn.Conv2d(in_channels,
                                        self.num_anchors * 4,
                                        kernel_size=1,
                                        stride=1)

    def generate_anchors(self, image, feat):
        grid_h, grid_w = feat.shape[-2:]
        image_h, image_w = image.shape[-2:]

        strid_h = torch.tensor(image_h // grid_h,
                               dtype=torch.int64,
                               device=feat.device)
        strid_h = torch.tensor(image_w // grid_w,
                               dtype=torch.int64,
                               device=feat.device)
        scale = torch.totensor(self.scales,
                               dtype=feat.dtype,
                               device=feat.device)
        aspect_ratio = torch.tensor(self.aspect_ratios,
                                    dtype=feat.dtype,
                                    device=feat.device)
        # below code ensure h/w = aspect_ratio, h*w = 1
        h_ratio = torch.sqrt(aspect_ratio)
        w_ratio = 1/h_ratio
        #     [3x1] * [1x3] -> [3x3].view(-1) -> len[9]
        # Get Box H and W
        ws = (w_ratio[:, None] * scale[:, None]).view(-1)
        hs = (h_ratio[:, None] * scale[:, None]).view(-1)

        base_anchor = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchor = base_anchor.round()

    def forward(self, image, feat, target):
        # Call RPN Layer
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        cls_score = self.cls_layer(rpn_feat)
        bbox_transform_pred = self.bbox_reg_layer(rpn_feat)
        # Generate Anchor
