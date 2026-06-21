import logging
from functools import partial
import torch
torch.cuda.empty_cache()  # 清理未使用的CUDA内存
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_

_logger = logging.getLogger(__name__)


class ChannelNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class DepthwiseConvBlock(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.depth_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = ChannelNorm(dim, eps=1e-6)
        self.point_conv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.point_conv2 = nn.Linear(4 * dim, dim)
        self.layer_scale = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        ) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = x.permute(0, 3, 1, 2)  # (B,H,W,C)→(B,C,H,W)
        x = self.depth_conv(x)
        x = x.permute(0, 2, 3, 1)  # (B,C,H,W)→(B,H,W,C)
        x = self.norm(x)
        x = self.point_conv1(x)
        x = self.act(x)
        x = self.point_conv2(x)
        if self.layer_scale is not None:
            x = self.layer_scale * x
        x = input + self.drop_path(x)
        return x


class FeatureTransformBlock(nn.Module):
    r"""深度可分离卷积分支的特征转换块"""
    def __init__(self, dim, drop_path=0., norm_layer=nn.LayerNorm, layer_scale_init_value=1e-6):
        super().__init__()
        self.norm = norm_layer(dim)
        self.depthwise_block = DepthwiseConvBlock(
            dim=dim, drop_path=drop_path, layer_scale_init_value=layer_scale_init_value
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.depthwise_block(self.norm(x)))
        return x


class FreqMLP(nn.Module):
    r"""频域MLP块（全局/频域特征分支核心）"""

    def __init__(self, dim, num_groups=1, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.num_groups = num_groups
        hidden_dim = int(dim * mlp_ratio)

        # 确保dim能被num_groups整除
        assert dim % num_groups == 0, f"dim {dim} must be divisible by num_groups {num_groups}"

        # 实部和虚部分享相同的MLP权重（也可设计独立权重）
        self.group_linear1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, groups=num_groups)
        self.act = nn.GELU()
        self.group_linear2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, groups=num_groups)

    def forward(self, x):
        # 输入形状：(B, H, W, C) → 转换为卷积所需的(B, C, H, W)
        x = x.permute(0, 3, 1, 2)  # (B,H,W,C) → (B,C,H,W)

        # 1. 实数→频域（保留完整复数信息）
        x_complex = torch.complex(x, torch.zeros_like(x))  # 虚部初始为0
        x_fft = torch.fft.fft2(x_complex)  # 输出：(B,C,H,W) 复数张量

        # 2. 分离实部和虚部，并行处理
        real_part = x_fft.real  # 实部：(B,C,H,W)
        imag_part = x_fft.imag  # 虚部：(B,C,H,W)

        # 3. 实部和虚部分别通过MLP
        real_part = self.group_linear1(real_part)
        real_part = self.act(real_part)
        real_part = self.group_linear2(real_part)

        imag_part = self.group_linear1(imag_part)
        imag_part = self.act(imag_part)
        imag_part = self.group_linear2(imag_part)

        # 4. 重组为复数张量
        x_fft_processed = torch.complex(real_part, imag_part)

        # 5. 频域→实数（逆FFT）
        x_ifft = torch.fft.ifft2(x_fft_processed)
        x = x_ifft.real  # 取实部（虚部理论上应为0，因输入是实数）

        # 6. 转换回原始形状：(B,C,H,W) → (B,H,W,C)
        x = x.permute(0, 2, 3, 1)

        return x


class FreqFeatureTransformBlock(nn.Module):
    r"""频域MLP分支的特征转换块"""
    def __init__(self, dim, num_groups=1, mlp_ratio=4.0, drop_path=0., norm_layer=nn.LayerNorm, layer_scale_init_value=1e-6):
        super().__init__()
        self.norm = norm_layer(dim)
        self.freq_mlp = FreqMLP(dim, num_groups=num_groups, mlp_ratio=mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        ) if layer_scale_init_value > 0 else None

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.freq_mlp(x)
        if self.layer_scale is not None:
            x = self.layer_scale * x
        x = residual + self.drop_path(x)
        return x


class InputProjection(nn.Module):
    r"""输入投影层（双分支共享输入预处理）"""
    def __init__(self, in_chans, embed_dim, kernel_size=3, stride=3, padding=1):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = ChannelNorm(embed_dim, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)  # (B,C_in,H,W)→(B,C_embed,H',W')
        x = self.norm(x)
        x = self.act(x)
        return x.permute(0, 2, 3, 1)  # (B,C_embed,H',W')→(B,H',W',C_embed)


class OutputProjection(nn.Module):
    r"""输出反投影层（适配双分支融合后的特征维度）"""
    def __init__(self, img_size, in_chans, embed_dim, kernel_size=8, stride=8):
        super().__init__()
        self.proj = nn.ConvTranspose2d(embed_dim, in_chans, kernel_size=kernel_size, stride=stride)
        self.norm = ChannelNorm(in_chans, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.img_size = img_size

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (B,H',W',C_fuse)→(B,C_fuse,H',W')
        x = self.proj(x)  # 反卷积恢复原始尺寸
        x = self.norm(x)
        x = self.act(x)
        return x


class DualBranchFeatureExtractor(nn.Module):
    r"""双分支特征提取器：深度可分离卷积（局部）+ 频域MLP（全局）并行"""
    def __init__(self, img_size=64, patch_size=8, in_chans=64, embed_dim=120, depths=3,
                 drop_path_rate=0.1, num_groups=15, mlp_ratio=4.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim  # 单个分支的嵌入维度
        self.fuse_dim = 2 * embed_dim  # 双分支拼接后维度（embed_dim*2）
        self.depths = depths  # 每个分支的块数量

        # 1. 共享输入投影层（双分支用同一预处理后的特征）
        self.input_proj = InputProjection(
            in_chans=in_chans,
            embed_dim=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=patch_size // 2
        )

        # 2. 位置嵌入（双分支共享，补充空间信息）
        self.H_patch = img_size // patch_size
        self.W_patch = img_size // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.H_patch, self.W_patch, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        # 3. 随机深度配置（双分支共享同一drop_path序列，保证训练一致性）
        self.drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        # 4. 构建双分支（并行的特征转换块序列）
        # 4.1 深度可分离卷积分支（局部特征）
        self.depthwise_branch = nn.Sequential(*[
            FeatureTransformBlock(
                dim=embed_dim,
                drop_path=self.drop_path_rates[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                layer_scale_init_value=1e-6
            ) for i in range(depths)
        ])

        # 4.2 频域MLP分支（全局/频域特征）
        self.freq_branch = nn.Sequential(*[
            FreqFeatureTransformBlock(
                dim=embed_dim,
                num_groups=num_groups,  # 分组数：默认embed_dim//8=120//8=15
                mlp_ratio=mlp_ratio,
                drop_path=self.drop_path_rates[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                layer_scale_init_value=1e-6
            ) for i in range(depths)
        ])

        # 5. 输出投影层（输入维度=融合后维度fuse_dim）
        self.output_proj = OutputProjection(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=self.fuse_dim,  # 关键：适配拼接后的维度
            kernel_size=patch_size,
            stride=patch_size
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, ChannelNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # 步骤1：共享输入预处理
        x = self.input_proj(x)  # (B,C_in,H,W)→(B,H_patch,W_patch,embed_dim)
        # 关键修改：动态调整pos_embed尺寸以匹配x的实际尺寸
        pos_embed = self.pos_embed.permute(0, 3, 1, 2)  # (1, C, H_pos, W_pos)
        pos_embed = F.interpolate(
            pos_embed,
            size=(x.shape[1], x.shape[2]),  # 匹配x的高度和宽度
            mode='bilinear',
            align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # 转回(B, H, W, C)

        # 添加位置嵌入（此时尺寸已匹配）
        x = x + pos_embed

        # 步骤2：双分支并行处理（输入相同，各自提取特征）
        x_depthwise = self.depthwise_branch(x)  # 深度可分离分支输出：(B,H_patch,W_patch,embed_dim)
        x_freq = self.freq_branch(x)            # 频域MLP分支输出：(B,H_patch,W_patch,embed_dim)

        # 步骤3：特征融合（通道拼接，dim=3对应channels_last的通道维度）
        x_fuse = torch.cat([x_depthwise, x_freq], dim=3)  # 融合后：(B,H_patch,W_patch,fuse_dim=2*embed_dim)

        # 步骤4：输出恢复（反投影到原始尺寸）
        x_out = self.output_proj(x_fuse)  # (B,H_patch,W_patch,fuse_dim)→(B,C_in,H,W)

        return x_out


# 双分支模型构建函数（方便直接调用）
def dual_branch_feat_extractor(patch_size=8, in_chans=180, img_size=64, depths=3, **kwargs):
    """构建双分支特征提取器实例（深度可分离卷积+频域MLP并行）"""
    return DualBranchFeatureExtractor(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=120,  # 单个分支嵌入维度
        depths=depths,  # 每个分支的块数量
        drop_path_rate=0.1,
        num_groups=15,  # 120//8=15，确保分组数能整除embed_dim
        mlp_ratio=4.0,** kwargs
    )