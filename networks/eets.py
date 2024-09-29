import torch.nn as nn
from .eets_main import SwinMain
from .utils import edge_conv2d3, SingleChannelConv

class EETS_First(nn.Module):
    def __init__(self, img_size=512,in_chans=3, num_classes=21843, window_size=7, zero_head=False, device=None):
        super(EETS_First, self).__init__()
        self.num_classes = num_classes
        self.attention = True
        self.zero_head = zero_head
        self.img_size = img_size
        self.patch_size = 4
        self.in_chans = in_chans
        self.window_size = window_size
        self.depths = [2, 2, 6, 2]
        self.num_heads = [3, 6, 12, 24]
        self.depths_decoder = [2, 2, 6, 2]
        self.embed_dim = 96
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.qk_scale = None
        self.drop_rate = 0.0
        self.drop_path_rate = 0.1
        self.ape = False
        self.patch_norm = True
        self.use_checkpoint = False
        self.device=device

        self.eets_first = SwinMain(img_size=self.img_size,
                                                     patch_size=self.patch_size,
                                                     in_chans=self.in_chans,
                                                     num_classes=self.num_classes,
                                                     embed_dim=self.embed_dim,
                                                     depths=self.depths,
                                                     depths_decoder=self.depths_decoder,
                                                     num_heads=self.num_heads,
                                                     window_size=self.window_size,
                                                     mlp_ratio=self.mlp_ratio,
                                                     qkv_bias=self.qkv_bias,
                                                     qk_scale=self.qk_scale,
                                                     drop_rate=self.drop_rate,
                                                     drop_path_rate=self.drop_path_rate,
                                                     ape=self.ape,
                                                     patch_norm=self.patch_norm,
                                                     use_checkpoint=self.use_checkpoint)
        self.getEdge = SingleChannelConv(3)

    def forward(self, x):
        x = x.to(self.device)
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        edge = edge_conv2d3(x)
        y = self.getEdge(edge)

        logits = self.eets_first(x, y)
        return logits



class EETS_Second(nn.Module):
    def __init__(self, img_size=1000, in_chans=3, num_classes=21843,window_size=7, zero_head=False, device='cuda'):
        super(EETS_Second, self).__init__()
        self.num_classes = num_classes
        self.attention = True
        self.zero_head = zero_head
        self.img_size = img_size
        self.patch_size = 4
        self.in_chans = in_chans
        self.window_size = window_size
        self.depths = [2, 2, 6, 2]
        self.num_heads = [3, 6, 12, 24]
        self.depths_decoder = [2, 2, 6, 2]
        self.embed_dim = 96
        self.mlp_ratio = 4.
        self.qkv_bias = True
        self.qk_scale = None
        self.drop_rate = 0.0
        self.drop_path_rate = 0.1
        self.ape = False
        self.patch_norm = True
        self.use_checkpoint = False
        self.device = device

        self.eets_second = SwinMain(img_size=self.img_size,
                                                        patch_size=self.patch_size,
                                                        in_chans=self.in_chans,
                                                        num_classes=self.num_classes,
                                                        embed_dim=self.embed_dim,
                                                        depths=self.depths,
                                                        depths_decoder=self.depths_decoder,
                                                        num_heads=self.num_heads,
                                                        window_size=self.window_size,
                                                        mlp_ratio=self.mlp_ratio,
                                                        qkv_bias=self.qkv_bias,
                                                        qk_scale=self.qk_scale,
                                                        drop_rate=self.drop_rate,
                                                        drop_path_rate=self.drop_path_rate,
                                                        ape=self.ape,
                                                        patch_norm=self.patch_norm,
                                                        use_checkpoint=self.use_checkpoint)


    def forward(self, x, y):

        x = x.to(self.device)
        y = y.to(self.device)

        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        y = y.unsqueeze(1)
        y_1 = y[:, 0:1, :, :].to(self.device)

        result = self.eets_second(x, y_1)
        return result


