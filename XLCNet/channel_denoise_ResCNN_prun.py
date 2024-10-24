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

############## training set ##################
data_num_train=100000
## load channel
H_noisy_in=zeros((data_num_train,Nx,Ny,2), dtype=float)
H_true_out=zeros((data_num_train,Nx,Ny,2), dtype=float)
# data1 = sio.loadmat('...\\Channel_f1n5_256ANTS_1000by100')
data1 = sio.loadmat('./data/Channel_f1n5_256ANTS_1000by100.mat')
channel = data1['Channel_mat']
for i in range(data_num_train):
    h = channel[i]
    H = np.reshape(h, (Nx,Ny))
    H_true_out[i, :, :, 0] = np.real(H)
    H_true_out[i, :, :, 1] = np.imag(H)
    noise = 1 / np.sqrt(2) * np.random.randn(Nx,Ny) + 1j * 1 / np.sqrt(2) * np.random.randn(
        Nx,Ny)
    H_noisy = H + 1 / np.sqrt(P) * noise
    H_noisy_in[i, :, :, 0] = np.real(H_noisy)
    H_noisy_in[i, :, :, 1] = np.imag(H_noisy)
print(((H_noisy_in)**2).mean(),((H_true_out)**2).mean())
print(H_noisy_in.shape,H_true_out.shape)


############## testing set ##################
data_num_test=2000
## load channel
H_noisy_in_test=zeros((data_num_test,Nx,Ny,2), dtype=float)
H_true_out_test=zeros((data_num_test,Nx,Ny,2), dtype=float)
# data1 = sio.loadmat('...\\Channel_f6n0_256ANTS_10by200')
data1 = sio.loadmat('./data/Channel_f1n5_256ANTS_10by200.mat')
channel = data1['Channel_mat']
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
modelpath = './model/ResCNN9_f1n5_256ANTS_1Kby100data_10dB_200ep_lr1e-3.hdf5'
ResCNN = load_model(modelpath)
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

# Weight Pruning for ResCNN
import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
batch_size = 128
epochs = 50
validation_split = 0.1

num_train = data_num_train * (1 - validation_split)
end_step = np.ceil(num_train / batch_size).astype(np.int32) * epochs
print(end_step)

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.8,
                                                               begin_step=0,
                                                               end_step=end_step)}

prune_model = prune_low_magnitude(ResCNN, **pruning_params)

# `prune_low_magnitude` requires a recompile.
prune_model.compile(optimizer='adam', loss='mse')

#prune_model.summary()

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

prune_model.fit(H_noisy_in, H_true_out,
                  batch_size=batch_size, epochs=epochs, validation_split=validation_split,
                  callbacks=callbacks, verbose=2, shuffle=True)


model_for_export = tfmot.sparsity.keras.strip_pruning(prune_model)
model_for_export.compile(optimizer='adam', loss='mse')

filepath = './model/ResCNN9_f1n5_256ANTS_1Kby100data_10dB_200ep_Prun08_50ep.hdf5'
tf.keras.models.save_model(model_for_export, filepath, include_optimizer=False)
print('Saved pruned Keras model to:', filepath)

PrunResCNN = load_model(filepath)
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