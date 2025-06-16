import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        #print(x.shape)
        x = self.fc2(self.fc1(x)) + x
        return x

from timm.models.layers import DropPath
from mamba_ssm import Mamba

class ASL(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.input_dim = in_channel
        self.output_dim = in_channel
        self.norm = nn.LayerNorm(in_channel)
        self.mamba = Mamba(
            d_model=self.input_dim // 2,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.proj = nn.Linear(in_channel, in_channel)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.convtout = nn.Conv2d(in_channel*2, in_channel, kernel_size=1)
        self.drop = DropPath(0.2)

        # 新增：两个可学习的标量权重
        self.weight_rgb = nn.Parameter(torch.tensor(1.0))
        self.weight_ir = nn.Parameter(torch.tensor(1.0))

    def forward(self, rgb, ir):
        ####### 分支1
        B, C = rgb.shape[:2]
        n_tokens = rgb.shape[2:].numel()
        img_dims = rgb.shape[2:]
        x_flat = rgb.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x1, x2 = torch.chunk(x_norm, 2, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba = torch.cat([x_mamba1, x_mamba2], dim=2)
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)

        ####### 分支2
        B2, C2 = ir.shape[:2]
        assert C == self.input_dim
        n_tokens2 = ir.shape[2:].numel()
        img_dims2 = ir.shape[2:]
        x_flat2 = ir.reshape(B2, C2, n_tokens2).transpose(-1, -2)
        x_norm2 = self.norm(x_flat2)
        x21, x22 = torch.chunk(x_norm2, 2, dim=2)
        x_mamba21 = self.mamba(x21) + self.skip_scale * x21   # 注意这里是 self.mamba 不是 self.x_mamba，代码原来写错了
        x_mamba22 = self.mamba(x22) + self.skip_scale * x22
        x_mamba2 = torch.cat([x_mamba21, x_mamba22], dim=2)
        x_mamba2 = self.norm(x_mamba2)
        x_mamba2 = self.proj(x_mamba2)
        out2 = x_mamba2.transpose(-1, -2).reshape(B, self.output_dim, *img_dims2)
        out2 = self.drop(out2)

        # 新增：权重乘法
        out = self.weight_rgb * out
        out2 = self.weight_ir * out2

        outnew = torch.cat([out, out2], dim=1)
        outnew = self.convtout(outnew)

        return outnew
