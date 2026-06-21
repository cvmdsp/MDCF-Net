import numpy as np
import scipy.io as sio
import os
import glob
import torch
import cv2
import Pypher
import random
import re

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dataparallel(model, ngpus, gpu0=None):
    if ngpus > 0:
        assert torch.cuda.device_count() >= ngpus
        gpu_list = list(range(gpu0, gpu0 + ngpus)) if gpu0 is not None else None
        if ngpus > 1:
            if not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model, gpu_list)
            else:
                model = model
        elif ngpus == 1:
            model = model
    else:
        model = model
    return model


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def para_setting(kernel_type,sf,sz,sigma):
    if kernel_type ==  'uniform_blur':
        psf = np.ones([sf,sf]) / (sf *sf)
    elif kernel_type == 'gaussian_blur':
        psf = np.multiply(cv2.getGaussianKernel(sf, sigma), (cv2.getGaussianKernel(sf, sigma)).T)

    fft_B = Pypher.psf2otf(psf, sz)
    fft_BT = np.conj(fft_B)
    return fft_B,fft_BT


def prepare_data(path, file_list, file_num):
    HR_HSI = np.zeros((((512,512,31,file_num))))
    HR_MSI = np.zeros((((512,512,3,file_num))))
    for idx in range(file_num):
        ####  read HR-HSI
        HR_code = file_list[idx]
        path1 = os.path.join(path, 'HSI/') + HR_code + '.mat'
        data = sio.loadmat(path1)
        HR_HSI[:,:,:,idx] = data['HR_HSI']

        ####  get HR-MSI
        path2 = os.path.join(path, 'RGB/') + HR_code + '.mat'
        data = sio.loadmat(path2)
        HR_MSI[:,:,:,idx] = data['HR_MSI']
    return HR_HSI, HR_MSI

def loadpath(pathlistfile,shuffle=True):
    fp = open(pathlistfile)
    pathlist = fp.read().splitlines()
    fp.close()
    if shuffle==True:
        random.shuffle(pathlist)
    return pathlist