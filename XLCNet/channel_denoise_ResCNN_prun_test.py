from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Conv2D, MaxPool2D, Add, Activation, Subtract, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from numpy import *
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #allow growth
import scipy.io as sio
import tempfile
import time

N=256 # BS antennas
snr=10
P=10**(snr/10.0)
scale=1
Nx = 16
Ny = 16


############## testing set ##################
data_num_test=2000
## load channel
H_noisy_in_test=zeros((data_num_test,Nx,Ny,2), dtype=float)
H_true_out_test=zeros((data_num_test,Nx,Ny,2), dtype=float)
data1 = sio.loadmat('./data/Channel_f1n5_near_256ANTS_10by200')
channel = data1['Channel_near_mat']
for i in range(data_num_test):
    h = channel[i]
    H = np.reshape(h, (Nx,Ny))
    H_true_out_test[i, :, :, 0] = np.real(H)
    H_true_out_test[i, :, :, 1] = np.imag(H)
    noise = 1 / np.sqrt(2) * np.random.randn(Nx,Ny) + 1j * 1 / np.sqrt(2) * np.random.randn(Nx,Ny)
    H_noisy = H + 1 / np.sqrt(P) * noise
    H_noisy_in_test[i, :, :, 0] = np.real(H_noisy)
    H_noisy_in_test[i, :, :, 1] = np.imag(H_noisy)

# load model
ResCNN = load_model('./model/ResCNN9_f1n5_256ANTS_1Kby100data_10dB_200ep_lr1e-3.hdf5')
ResCNN.summary()
decoded_channel = ResCNN.predict(H_noisy_in_test)
nmse1=zeros((data_num_test,1), dtype=float)
nmse2=zeros((data_num_test,1), dtype=float)
for n in range(data_num_test):
    MSE1 = ((H_true_out_test[n,:,:,:] - H_noisy_in_test[n,:,:,:]) ** 2).sum()
    MSE2=((H_true_out_test[n,:,:,:]-decoded_channel[n,:,:,:])**2).sum()
    norm_real=((H_true_out_test[n,:,:,:])**2).sum()
    nmse1[n] = MSE1 / norm_real
    nmse2[n]=MSE2/norm_real
print(nmse1.sum()/(data_num_test), nmse2.sum()/(data_num_test))

PrunResCNN = load_model('./model/ResCNN9_f1n5_256ANTS_1Kby100data_10dB_200ep_Prun08_50ep.hdf5')
PrunResCNN.summary()
# 获取模型的所有权重
weights = PrunResCNN.get_weights()
# 初始化计数器
zero_count = 0
nonzero_count = 0
# 遍历权重
for weight in weights:
    # 计算每个权重中0的数量
    zero_count += np.count_nonzero(weight == 0)
    nonzero_count += np.count_nonzero(weight)
#print(f"模型中参数为0的个数是: {zero_count}")
print(f"模型中参数非0的个数是: {nonzero_count}")
print(f"模型中参数非0的比例是: {nonzero_count/(nonzero_count+zero_count)}")

decoded_channel = PrunResCNN.predict(H_noisy_in_test)
nmse1=zeros((data_num_test,1), dtype=float)
nmse2=zeros((data_num_test,1), dtype=float)
for n in range(data_num_test):
    MSE1 = ((H_true_out_test[n,:,:,:] - H_noisy_in_test[n,:,:,:]) ** 2).sum()
    MSE2=((H_true_out_test[n,:,:,:]-decoded_channel[n,:,:,:])**2).sum()
    norm_real=((H_true_out_test[n,:,:,:])**2).sum()
    nmse1[n] = MSE1 / norm_real
    nmse2[n]=MSE2/norm_real
print(nmse1.sum()/(data_num_test), nmse2.sum()/(data_num_test))