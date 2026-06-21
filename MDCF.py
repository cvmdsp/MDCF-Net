import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from DC import dual_branch_feat_extractor


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:
            x = self.relu(x)
        return x


class EnhancedCBR(nn.Module):
    """增强版CBR - 串行多尺度特征提取"""

    def __init__(self, in_c, out_c):
        super().__init__()
        # 关键改进1: 使用串行结构而非并行，避免信息分割
        # 第一层: 标准3x3卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        # 第二层: 空洞卷积增加感受野
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        # 残差连接
        self.residual = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False) if in_c != out_c else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out + identity


class DCPFM(nn.Module):


    def __init__(self, in_c, dim, num_heads=8, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5


        self.v = nn.Linear(dim, dim)


        self.channel_attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )


        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)


        self.input_cbr = nn.Sequential(
            EnhancedCBR(in_c, dim),  # 串行多尺度
            CBR(dim, dim, kernel_size=3, padding=1)  # 标准卷积
        )


        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            nn.Conv2d(dim, dim, kernel_size=1)  # 1x1卷积降低计算量
        )

    def compute_attention(self, feature_map, B, H, W, C, feature_type):

        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)



        channel_weight = feature_map_pooled.mean(dim=[1, 2])  # [B, C]
        channel_weight = self.channel_attn(channel_weight).unsqueeze(1).unsqueeze(1)  # [B, 1, 1, C]
        feature_map_enhanced = feature_map_pooled * channel_weight  # 通道加权

        # 标准注意力计算
        attn = attn_layer(feature_map_enhanced).reshape(
            B, h * w, self.num_heads,
               self.kernel_size * self.kernel_size,
               self.kernel_size * self.kernel_size
        ).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        return self.attn_drop(attn)

    def apply_attention(self, attn, v, B, H, W, C):

        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1
        )
        x_weighted = F.fold(
            x_weighted,
            output_size=(H, W),
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride
        )
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        return self.proj_drop(x_weighted)

    def forward(self, x, fg, bg):
        x = self.input_cbr(x)
        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape
        v = self.v(x).permute(0, 3, 1, 2)


        v_unfolded = self.unfold(v).reshape(
            B, self.num_heads, self.head_dim,
            self.kernel_size * self.kernel_size, -1
        ).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')
        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)


        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(
            B, self.num_heads, self.head_dim,
            self.kernel_size * self.kernel_size, -1
        ).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')
        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)

        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)
        return self.output_cbr(x_weighted_bg)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(residual)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        return out


class Future(nn.Module):
    def __init__(self, n_chans, n_feats):
        super(Future, self).__init__()
        assert n_chans > 0, f"输入通道数必须大于0，但得到n_chans={n_chans}"

        self.res_blocks = 2

        self.weight_a = nn.Parameter(torch.ones(1))
        self.weight_b = nn.Parameter(torch.ones(1))
        self.weight_c = nn.Parameter(torch.ones(1))
        self.weight_d = nn.Parameter(torch.ones(1))

        self.h2 = nn.Conv2d(n_chans, n_feats, kernel_size=3, padding=1)

        self.feature = nn.ModuleList([ResBlock(n_feats, n_feats) for _ in range(self.res_blocks)])

        self.it1 = dual_branch_feat_extractor(patch_size=8, in_chans=n_feats)
        self.it2 = dual_branch_feat_extractor(patch_size=4, in_chans=n_feats)
        self.it3 = dual_branch_feat_extractor(patch_size=16, in_chans=n_feats)

        self.DCPF = DCPFM(
            in_c=n_feats,
            dim=n_feats,
            num_heads=6,
            kernel_size=3,
            stride=1
        )

    def forward(self, x):
        y = self.h2(x)

        for i in range(self.res_blocks):
            y = self.feature[i](y)

        y1 = self.it1(y)
        y2 = self.it2(y)
        y3 = self.it3(y)

        target_size = (y.shape[2], y.shape[3])
        y1 = F.interpolate(y1, size=target_size, mode='bilinear', align_corners=False)
        y2 = F.interpolate(y2, size=target_size, mode='bilinear', align_corners=False)
        y3 = F.interpolate(y3, size=target_size, mode='bilinear', align_corners=False)

        total_weight = self.weight_a + self.weight_b + self.weight_c + self.weight_d
        y_body = (y1 * (self.weight_a / total_weight) +
                  y2 * (self.weight_b / total_weight) +
                  y3 * (self.weight_c / total_weight))

        y_base_fused = y * (self.weight_d / total_weight) + y_body
        y_aggregated = self.DCPF(x=y_base_fused, fg=y1, bg=y3)
        y_final = y + y_body + y_aggregated

        return y_final


class MDCFNET(nn.Module):
    def __init__(self, hsi_chans=31, msi_chans=3, scale_factor=4):
        super(MDCFNET, self).__init__()
        self.hsi_chans = hsi_chans
        self.msi_chans = msi_chans
        self.scale_factor = scale_factor
        self.input_channels = hsi_chans + msi_chans

        self.MPDFE= Future(n_chans=self.input_channels, n_feats=180)

        self.refine = nn.Sequential(
            nn.Conv2d(180, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, hsi_chans, 3, 1, 1)
        )

    def forward(self, HSI, MSI):
        UP_LRHSI = F.interpolate(HSI, scale_factor=self.scale_factor, mode='bicubic')
        UP_LRHSI = UP_LRHSI.clamp_(0, 1)

        target_h, target_w = UP_LRHSI.shape[2], UP_LRHSI.shape[3]
        MSI_upsampled = F.interpolate(
            MSI,
            size=(target_h, target_w),
            mode='bicubic',
            align_corners=False
        )

        Data = torch.cat((UP_LRHSI, MSI_upsampled), dim=1)
        Data = self.MPDFE(Data)

        output = self.refine(Data)
        output = output + UP_LRHSI
        output = output.clamp_(0, 1)

        return output