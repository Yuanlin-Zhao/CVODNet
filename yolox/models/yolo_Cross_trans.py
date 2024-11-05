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
class MutilTransformer(nn.Module):
    def __init__(self, in_channel):

        super().__init__()
        self.input_dim = in_channel
        self.output_dim = in_channel
        self.norm = nn.LayerNorm(in_channel)
        self.transformer = TransformerLayer(
            c=in_channel // 2,  # Model dimension d_model
            num_heads=8
        )

        self.proj = nn.Linear(in_channel, in_channel)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.convtout = nn.Conv2d(in_channel*2, in_channel, kernel_size=1)
        self.drop = DropPath(0.2)

    def forward(self, rgb, ir):
        #######分支1
        #print(f'self.input_dim={self.input_dim}')
        #print(rgb.shape)
        B, C = rgb.shape[:2]
        #assert C == self.input_dim
        n_tokens = rgb.shape[2:].numel()
        img_dims = rgb.shape[2:]
        x_flat = rgb.reshape(B, C, n_tokens).transpose(-1, -2)
        #print(x_flat.shape)

        x_norm = self.norm(x_flat)
        x1, x2 = torch.chunk(x_norm, 2, dim=2)

        x_transformer1 = self.transformer(x1) + self.skip_scale * x1
        x_transformer2 = self.transformer(x2) + self.skip_scale * x2

        x_transformer = torch.cat([x_transformer1, x_transformer2], dim=2)
        x_transformer = self.norm(x_transformer)
        x_transformer = self.proj(x_transformer)
        out = x_transformer.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)

        #######分支2
        B2, C2 = ir.shape[:2]
        assert C == self.input_dim
        n_tokens2 = ir.shape[2:].numel()
        img_dims2 = ir.shape[2:]
        x_flat2 = ir.reshape(B2, C2, n_tokens2).transpose(-1, -2)
        x_norm2 = self.norm(x_flat2)
        x21, x22  = torch.chunk(x_norm2, 2, dim=2)
        x_transformer21 = self.transformer(x21) + self.skip_scale * x21
        x_transformer22 = self.transformer(x22) + self.skip_scale * x22

        x_transformer2 = torch.cat([x_transformer21, x_transformer22], dim=2)
        x_transformer2 = self.norm(x_transformer2)
        x_transformer2 = self.proj(x_transformer2)
        out2 = x_transformer2.transpose(-1, -2).reshape(B, self.output_dim, *img_dims2)
        out2 = self.drop(out2)
        outnew = torch.cat([out, out2], dim=1)
        outnew = self.convtout(outnew)

        return outnew
