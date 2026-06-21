# import scipy
from MDCF import MDCFNET
from cave_Dataset import cave_dataset
import torch.utils.data as tud
import time
import datetime
import argparse
from torch.autograd import Variable
from Utils import *
import torch.nn as nn
import os

# 关键修改：设置matplotlib非交互式后端，避免GUI错误
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot前设置
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    class criterion0(nn.Module):
        def __init__(self):
            super(criterion0, self).__init__()
            self.loss1 = nn.L1Loss()

        def forward(self, x, y):
            loss1 = self.loss1(x, y)
            return loss1


    # Model Config
    parser = argparse.ArgumentParser(description="PyTorch Code for HSI Fusion")
    parser.add_argument('--data_path', default='./Data/Train/', type=str, help='Path of the training data')
    parser.add_argument("--sizeI", default=64, type=int, help='The image size of the training patches')
    parser.add_argument("--batch_size", default=16, type=int, help='Batch size')
    parser.add_argument("--trainset_num", default=20000, type=int, help='The number of training samples of each epoch')
    parser.add_argument("--sf", default=4, type=int, help='Scaling factor')
    opt = parser.parse_args()

    def seed_torch(seed=745104):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    seed_torch()
    print(opt)

    print("===> New Model")
    model = MDCFNET(31, 3, 4)

    # set the number of parallel GPUs
    print("===> Setting GPU")
    model = dataparallel(model, 1)

    # Initialize weightResult
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)

    # Load training data
    key = 'Train.txt'
    file_path = opt.data_path + key
    file_list = loadpath(file_path)
    HR_HSI, HR_MSI = prepare_data(opt.data_path, file_list, 20)

    # Load trained model
    initial_epoch = findLastCheckpoint(save_dir="Checkpoint/f4")
    if initial_epoch > 0:
        print('resuming by loading epoch %04d' % initial_epoch)
        model = torch.load(os.path.join("Checkpoint/f4", 'model_%04d.pth' % initial_epoch)).to(device)

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)

    # 初始化参数
    best_loss = float('inf')
    save_dir = "Checkpoint/f4"
    saved_models = []
    max_keep = 10
    train_losses = []  # 记录每轮损失

    # pipline of training
    for epoch in range(initial_epoch, 160):
        model.train().to(device)

        dataset = cave_dataset(opt, HR_HSI, HR_MSI)
        loader_train = tud.DataLoader(dataset, num_workers=1, batch_size=opt.batch_size, shuffle=True)

        epoch_loss = 0
        start_time = time.time()

        for i, (LR, RGB, HR) in enumerate(loader_train):
            LR, RGB, HR = Variable(LR), Variable(RGB), Variable(HR)
            out = model(LR.to(device), RGB.to(device)).to(device)

            criterion = criterion0()
            loss = criterion(out.to(device), HR.to(device))

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('%4d %4d / %4d loss = %.10f time = %s' % (
                epoch + 1, i, len(dataset) // opt.batch_size, epoch_loss / ((i + 1) * opt.batch_size),
                datetime.datetime.now()))

        elapsed_time = time.time() - start_time
        avg_loss = epoch_loss / len(dataset)
        train_losses.append(avg_loss)  # 记录当前轮平均损失
        print('epcoh = %4d , loss = %.10f , time = %4.2f s' % (epoch + 1, avg_loss, elapsed_time))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存当前轮模型
        current_model_path = os.path.join(save_dir, f'model_{(epoch + 1):04d}.pth')
        torch.save(model, current_model_path)
        saved_models.append(current_model_path)

        # 保留最后10轮模型
        if len(saved_models) > max_keep:
            oldest_model = saved_models.pop(0)
            if os.path.exists(oldest_model):
                os.remove(oldest_model)
                print(f"Removed old model: {oldest_model}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_dir, 'model_best.pth')
            torch.save(model, best_model_path)
            print(f"Best model updated: {best_model_path} (loss: {best_loss:.10f})")

    # 绘制并保存损失曲线
    def plot_loss_curve(losses, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2, label='Training Loss')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('L1 Loss', fontsize=14)
        plt.title('Training Loss Curve', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图像，释放内存

    # 绘制损失曲线
    plot_loss_curve(train_losses, os.path.join(save_dir, 'loss_curve.png'))
    print(f"Loss curve saved to: {os.path.join(save_dir, 'loss_curve.png')}")

    # 保存损失值到txt文件
    loss_file = os.path.join(save_dir, 'losses.txt')
    with open(loss_file, 'w') as f:
        for epoch, loss in enumerate(train_losses, 1):
            f.write(f"{epoch} {loss:.10f}\n")
    print(f"Loss values saved to: {loss_file}")

    print(f"Training completed. Saved:")
    print(f"- Best model: model_best.pth (loss: {best_loss:.10f})")
    print(f"- Last {max_keep} models: {[os.path.basename(p) for p in saved_models]}")