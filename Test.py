import os
import numpy as np
import torch
import torch.utils.data as tud
import argparse
from Utils import *
from cave_Dataset import cave_dataset
from thop import profile
import scipy.io
from MDCF import MDCFNET

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
parser.add_argument('--data_path', default='./Data/Test/', type=str, help='path of the testing data')
parser.add_argument("--sizeI", default=512, type=int, help='the size of trainset')
parser.add_argument("--testset_num", default=1, type=int, help='total number of testset')
parser.add_argument("--batch_size", default=1, type=int, help='Batch size')
parser.add_argument("--sf", default=4, type=int, help='Scaling factor')
parser.add_argument("--kernel_type", default='gaussian_blur', type=str, help='Kernel type')
opt = parser.parse_args()
print(opt)

key = 'Test.txt'
file_path = opt.data_path + key
file_list = loadpath(file_path, shuffle=False)
HR_HSI, HR_MSI = prepare_data(opt.data_path, file_list, 1)

dataset = cave_dataset(opt, HR_HSI, HR_MSI, istrain=False)
loader_train = tud.DataLoader(dataset, batch_size=opt.batch_size)

# ========== 修改后的模型加载部分 ==========
# 初始化新模型
model = MDCFNET(hsi_chans=31, msi_chans=3, scale_factor=opt.sf).to(device)
# 读取转换好的权重字典
ckpt = torch.load("./Checkpoint/f4/MDCF.pth", map_location=device)
# 载入参数
model.load_state_dict(ckpt["state_dict"])
# 推理模式
model.eval()
# ==========================================

result_dir = './Result/f4/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 先取一组数据用于计算flops
LR_sample, RGB_sample, _ = next(iter(loader_train))
flops, params = profile(model, inputs=(LR_sample.to(device), RGB_sample.to(device),))

for j, (LR, RGB, HR) in enumerate(loader_train):
    with torch.no_grad():
        out = model(LR.to(device), RGB.to(device))
        result = out
        result = result.clamp(min=0., max=1.)
        result_np = result.cpu().detach().numpy()
        result_combined = np.concatenate(result_np, axis=1)
        scipy.io.savemat(f'./Result/f4/'+file_list[j]+'.mat', {'result': result_combined})
