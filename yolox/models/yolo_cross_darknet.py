#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from torch import nn
import torch
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

from torchvision.transforms import ToPILImage

from yolox.models.yolo_Cross_trans import ASL

class TConv(nn.Module):
    def __init__(self, in_c, out_c, k, p, ):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=k, padding=p,groups=in_c)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def extract_middle_and_edges(x):

        B = x.size(0)
        assert B >= 8

        # 中间 4 帧
        mid_start = B // 2 - 2
        mid4 = x[mid_start:mid_start + 4]

        # 两边 4 帧（前2 + 后2）
        edge4 = torch.cat([x[:2], x[-2:]], dim=0)

        return mid4, edge4

class TIC(nn.Module):
    def __init__(self, in_c, out_C):
        super().__init__()

        self.conv_all1 = TConv(in_c * 8, out_C, k=3, p=1)
        self.pool_all = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv_all2 = TConv(out_C, out_C, k=3, p=1)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)

        self.conv_part1 = TConv(in_c * 4, out_C, k=3, p=1)
        self.conv_part2 = TConv(out_C, out_C, k=3, p=1)

        # 可学习权重矩阵，按通道设置，每个 [1, C, 1] 会自动 broadcast
        self.weight_all = nn.Parameter(torch.ones(1, out_C, 1))   # 用于 all8
        self.weight_mid = nn.Parameter(torch.ones(1, out_C, 1))   # 用于 mid4
        self.weight_edge = nn.Parameter(torch.ones(1, out_C, 1))  # 用于 edge4

    def forward(self, x):
        B, C, H, W = x.size()
        print(x.size())
        all = x.reshape(1, B * C, H * W)  # 合并为 [1, B*C, H*W]

        # 中间4帧
        mid_start = B // 2 - 2
        mid4 = x[mid_start:mid_start + 4]  # [4, C, H, W]
        edge4 = torch.cat([x[:2], x[-2:]], dim=0)  # [4, C, H, W]
        mid4 = mid4.reshape(1, B * C//2, H * W)  # 合并为 [1, B*C, H*W]
        edge4 = edge4.reshape(1, B * C//2, H * W)  # 合并为 [1, B*C, H*W]
        # --- all 路径 ---
        all = self.conv_all1(all)                   # [1, out_C, H*W]
        all = self.pool_all(all)                    # ↓ 长度减半
        all = self.conv_all2(all)
        all = self.up(all)                          # ↑ 恢复原始长度
        all8 = self.conv_all2(all)                  # [1, out_C, H*W]

        # --- mid4 路径 ---
        mid4 = self.conv_part1(mid4)
        mid4 = self.pool_all(mid4)
        mid4 = self.conv_part2(mid4)
        mid4 = self.up(mid4)
        mid4 = self.conv_part2(mid4)

        # --- edge4 路径 ---
        edge4 = self.conv_part1(edge4)
        edge4 = self.pool_all(edge4)
        edge4 = self.conv_part2(edge4)
        edge4 = self.up(edge4)
        edge4 = self.conv_part2(edge4)

        w_all = self.weight_all  # [1, C, 1]
        w_mid = self.weight_mid  # [1, C, 1]
        w_edge = self.weight_edge  # [1, C, 1]

        out = w_all * all8 * w_mid * mid4 * w_edge * edge4  # 逐通道广播乘法
        out.reshape(B, C, H, W)

        return out



class Cross_CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Conv(3, base_channels, 3, 2)
        self.stemir = Conv(3, base_channels, 3, 2)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )



        # dark2
        self.dark2ir = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3ir = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4ir = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5ir = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

        self.cross1 = ASL(base_channels * 4)
        self.cross2 = ASL(base_channels * 8)
        self.cross3 = ASL(base_channels * 16)
        self.dark3conv=nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1)
        self.dark4conv = nn.Conv2d(base_channels * 16, base_channels * 8, kernel_size=3, padding=1)
        self.dark5conv = nn.Conv2d(base_channels * 32, base_channels * 16, kernel_size=3, padding=1)

    def forward(self, rgb, ir):

        rgb = self.stem(rgb)

        rgbstem = rgb
        print(f'rgbstem={rgbstem.shape}')
        rgb = self.dark2(rgb)
        rgbsdark2 = rgb
        print(f'rgbsdark2={rgbsdark2.shape}')
        rgb = self.dark3(rgb)
        rgbsdark3 = rgb
        print(f'rgbsdark3={rgbsdark3.shape}')
        rgb = self.dark4(rgb)
        rgbsdark4 = rgb
        print(f'rgbsdark4={rgbsdark4.shape}')
        rgb = self.dark5(rgb)
        rgbsdark5 = rgb
        print(f'rgbsdark5={rgbsdark5.shape}')

        #红外

        ir = self.stemir(ir)
        irstem = ir
        ir = self.dark2ir(ir)
        irsdark2 = ir
        ir = self.dark3ir(ir)
        irsdark3 = ir
        ir = self.dark4ir(ir)
        irsdark4 = ir
        ir = self.dark5ir(ir)
        irsdark5 = ir
        #print(f'ir.shape={ir.shape}')
        # outputsdark3 = torch.cat([rgbsdark3, irsdark3], dim=1)
        # outputsdark3 = self.dark3conv(outputsdark3)
        # outputsdark4 = torch.cat([rgbsdark4, irsdark4], dim=1)
        # outputsdark4 = self.dark4conv(outputsdark4)
        # outputsdark5 = torch.cat([rgbsdark5, irsdark5], dim=1)
        # outputsdark5 = self.dark5conv(outputsdark5)
        outputsdark3 = self.cross1(rgbsdark3, irsdark3)
        outputsdark4 = self.cross2(rgbsdark4, irsdark4)
        outputsdark5 = self.cross3(rgbsdark5, irsdark5)
        outputsdark3=TIC(outputsdark3)
        outputsdark4=TIC(outputsdark4)
        outputsdark5=TIC(outputsdark5)

        return outputsdark3, outputsdark4, outputsdark5



