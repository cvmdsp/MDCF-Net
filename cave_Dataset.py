import torch.utils.data as tud
# import random
# import cv2
# import numpy as np
from Utils import *

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2


    def rfft(x, d):
        t = rfft2(x, dim=(-d))
        return torch.stack((t.real, t.imag), -1)


    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:, :, 0], x[:, :, 1]), s=signal_sizes, dim=(-d))


class cave_dataset(tud.Dataset):
    def __init__(self, opt, HR_HSI, HR_MSI, istrain=True):
        super(cave_dataset, self).__init__()
        self.path = opt.data_path
        self.istrain = istrain
        self.factor = opt.sf
        # 获取实际数据的第3维长度，用于索引边界检查
        self.data_dim3 = HR_HSI.shape[3] if HR_HSI is not None else 0

        if istrain:
            self.num = opt.trainset_num
            # 注意：self.file_num应与实际数据维度匹配，这里保留但不再用于索引生成
            self.file_num = 20
            self.sizeI = opt.sizeI
        else:
            self.num = opt.testset_num
            self.file_num = 12
            self.sizeI = 512

        self.HR_HSI, self.HR_MSI = HR_HSI, HR_MSI

        # 初始化时检查数据维度与配置是否匹配
        if self.istrain:
            assert self.data_dim3 > 0, "HR_HSI数据为空"
            # 如果实际数据量小于配置的file_num，给出警告
            if self.data_dim3 < self.file_num:
                print(f"警告：实际数据第3维长度({self.data_dim3})小于配置的file_num({self.file_num})")

    def __getitem__(self, index):
        if self.istrain:
            # 训练模式：基于实际数据维度生成随机索引，确保不越界
            index1 = random.randint(0, self.data_dim3 - 1)
        else:
            # 测试模式：使用取模确保索引有效
            index1 = index % self.data_dim3

        # 强制安全检查，发现问题立即报错
        assert index1 < self.data_dim3, \
            f"index1={index1} 越界，数据第3维大小为{self.data_dim3}"

        sigma = 2.0
        # 现在索引不会越界
        HR_HSI = self.HR_HSI[:, :, :, index1]
        HR_MSI = self.HR_MSI[:, :, :, index1]

        sz = [self.sizeI, self.sizeI]
        fft_B, fft_BT = para_setting('gaussian_blur', self.factor, sz, sigma)
        fft_B = torch.cat((torch.Tensor(np.real(fft_B)).unsqueeze(2),
                           torch.Tensor(np.imag(fft_B)).unsqueeze(2)), 2)
        fft_BT = torch.cat((torch.Tensor(np.real(fft_BT)).unsqueeze(2),
                            torch.Tensor(np.imag(fft_BT)).unsqueeze(2)), 2)

        # 确保随机裁剪不越界
        max_coord = 512 - self.sizeI
        assert max_coord >= 0, f"裁剪尺寸{self.sizeI}大于512"
        px = random.randint(0, max_coord)
        py = random.randint(0, max_coord)
        hr_hsi = HR_HSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]
        hr_msi = HR_MSI[px:px + self.sizeI:1, py:py + self.sizeI:1, :]

        if self.istrain:
            # 数据增强
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)

            # 随机旋转
            for _ in range(rotTimes):
                hr_hsi = np.rot90(hr_hsi)
                hr_msi = np.rot90(hr_msi)

            # 随机垂直翻转
            if vFlip:
                hr_hsi = hr_hsi[:, ::-1, :].copy()
                hr_msi = hr_msi[:, ::-1, :].copy()

            # 随机水平翻转
            if hFlip:
                hr_hsi = hr_hsi[::-1, :, :].copy()
                hr_msi = hr_msi[::-1, :, :].copy()

        # 生成低分辨率HSI
        lr_hsi = cv2.GaussianBlur(hr_hsi, (5, 5), 2)
        lr_hsi = cv2.resize(lr_hsi, (sz[0] // self.factor, sz[1] // self.factor))

        # 调整维度顺序并转换为Tensor
        hr_hsi = hr_hsi.copy().transpose(2, 0, 1)
        hr_msi = hr_msi.copy().transpose(2, 0, 1)
        lr_hsi = lr_hsi.copy().transpose(2, 0, 1)

        hr_hsi = torch.FloatTensor(hr_hsi)
        hr_msi = torch.FloatTensor(hr_msi)
        lr_hsi = torch.FloatTensor(lr_hsi)

        return lr_hsi, hr_msi, hr_hsi

    def __len__(self):
        return self.num