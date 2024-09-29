import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm, MultiheadAttention
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Mlp(nn.Module):
    def __init__(self,hidden_size=3,dropout_rate=0.1):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, 4*hidden_size)
        self.fc2 = Linear(4*hidden_size, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, num_heads, hidden_size, vis=None):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_dropout_rate = 0.0
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(self.attention_dropout_rate)
        self.proj_dropout = Dropout(self.attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights





class MSABlock(nn.Module):
    def __init__(self,hidden_size,num_heads, device='cuda', vis=False):
        super(MSABlock, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp(self.hidden_size)
        self.attn = Attention(self.num_heads, self.hidden_size)


    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, _ = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h


        return x


class EEBlock(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv = nn.Conv2d(4, 128, kernel_size=patch_size, stride=patch_size, groups= in_chans)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=96, kernel_size=(3, 3), stride=1, padding=1, groups=32)

        self.msa_layer = MSABlock(hidden_size=96, num_heads=8)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, y):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = torch.concat([x, y],dim=1)
        x = self.conv(x)
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)

        if self.norm is not None:
            x = self.norm(x)

        x = self.msa_layer(x)

        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops




def edge_conv2d3(im):

    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # print(sobel_kernel.shape)
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
    # print(sobel_kernel.shape)
    sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)
    # print(sobel_kernel.shape)
    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    edge_detect = torch.abs(conv_op(Variable(im)))

    conv_op1 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel1 = sobel_kernel1.reshape((1, 1, 3, 3))
    sobel_kernel1 = np.repeat(sobel_kernel1, 3, axis=1)
    sobel_kernel1 = np.repeat(sobel_kernel1, 3, axis=0)
    conv_op1.weight.data = torch.from_numpy(sobel_kernel1).cuda()
    edge_detect1 = torch.abs(conv_op1(Variable(im)))

    conv_op2 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel2 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel2 = sobel_kernel2.reshape((1, 1, 3, 3))
    sobel_kernel2 = np.repeat(sobel_kernel2, 3, axis=1)
    sobel_kernel2 = np.repeat(sobel_kernel2, 3, axis=0)
    conv_op2.weight.data = torch.from_numpy(sobel_kernel2).cuda()
    edge_detect2 = torch.abs(conv_op2(Variable(im)))

    conv_op3 = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    sobel_kernel3 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel3 = sobel_kernel3.reshape((1, 1, 3, 3))
    sobel_kernel3 = np.repeat(sobel_kernel3, 3, axis=1)
    sobel_kernel3 = np.repeat(sobel_kernel3, 3, axis=0)
    conv_op3.weight.data = torch.from_numpy(sobel_kernel3).cuda()
    edge_detect3 = torch.abs(conv_op3(Variable(im)))


    sobel_out = edge_detect+edge_detect1+edge_detect2+edge_detect3

    return sobel_out

class SingleChannelConv(nn.Module):
    def __init__(self, in_channels):
        super(SingleChannelConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)  # 使用 1x1 卷积核

        # 固定卷积核为均值卷积核
        self.conv.weight.data.fill_(1.0 / in_channels)

    def forward(self, x):
        return self.conv(x)