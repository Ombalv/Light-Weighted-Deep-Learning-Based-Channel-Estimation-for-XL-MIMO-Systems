from keras.layers import Input, Dense, Dropout, Conv1D, Conv2D, MaxPool2D, BatchNormalization, Add, Activation, Subtract, Flatten
from keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from numpy import *
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #allow growth
import scipy.io as sio

N=512 # BS antennas
snr=10
P=10**(snr/10.0)
scale=1
Nx = 16
Ny = 32

############## training set ##################
data_num_train=100000
## load channel
H_noisy_in=zeros((data_num_train,Nx,Ny,2), dtype=float)
H_true_out=zeros((data_num_train,Nx,Ny,2), dtype=float)
# data1 = sio.loadmat('...\\Channel_f1n5_256ANTS_1000by100')
data1 = sio.loadmat('data\Channel_f1n5_512ANTS_1000by100.mat')
channel = data1['Channel_mat']

# convert 100000x256 to 100000x16x16x2, 256=16x16, and separating real and imaginary parts into 2 channels
for i in range(data_num_train):
    h = channel[i]
    H = np.reshape(h, (Nx,Ny))
    H_true_out[i, :, :, 0] = np.real(H)
    H_true_out[i, :, :, 1] = np.imag(H)
    # add noise to the channel, noise is complex Gaussian noise
    noise = 1 / np.sqrt(2) * np.random.randn(Nx,Ny) + 1j * 1 / np.sqrt(2) * np.random.randn(Nx,Ny)
    H_noisy = H + 1 / np.sqrt(P) * noise
    H_noisy_in[i, :, :, 0] = np.real(H_noisy)
    H_noisy_in[i, :, :, 1] = np.imag(H_noisy)
print("H_noisy_in mean:",((H_noisy_in)**2).mean(),"\n","H_true_out mean:",((H_true_out)**2).mean())
print("Shape of H_noisy_in:", H_noisy_in.shape)
print("Shape of H_true_out:", H_true_out.shape)

## Build DNN model
K=3
input_dim = (Nx,Ny,2)
output_dim = 2
inp = Input(shape=input_dim)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(inp)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(xn)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(xn)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(xn)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(xn)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(xn)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(xn)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=64, kernel_size=(K,K), padding='Same', activation='relu')(xn)
xn = BatchNormalization()(xn)
xn = Conv2D(filters=output_dim, kernel_size=(K,K), padding='Same', activation='linear')(xn)
x1 = Subtract()([inp, xn])

model = Model(inputs=inp, outputs=x1)

# checkpoint
filepath='./model/ResCNN9_f1n5_512ANTS_1Kby100data_10dB_200ep_lr1e-3.hdf5'

# adam=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08) # original learning rate 1e-3
adam=Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss='mse')
model.summary()

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
callbacks_list = [checkpoint] # original callbacks_list
# callbacks_list = [checkpoint, early_stopping]

model.fit(x=H_noisy_in, y=H_true_out, epochs=200, batch_size=128, callbacks=callbacks_list, verbose=2, shuffle=True, validation_split=0.1)

############## testing set ##################
data_num_test=2000
## load channel
H_noisy_in_test=zeros((data_num_test,Nx,Ny,2), dtype=float)
H_true_out_test=zeros((data_num_test,Nx,Ny,2), dtype=float)
# data1 = sio.loadmat('...\\Channel_f0n6_256ANTS_10by200')
data1 = sio.loadmat('data\Channel_f1n5_512ANTS_10by200.mat')
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
# ResCNN2d = load_model('./model/ResCNN9_f1n5_256ANTS_1Kby100data_20dB_200ep_earlystop.hdf5')
ResCNN2d = load_model(filepath)

decoded_channel = ResCNN2d.predict(H_noisy_in_test)

print("Mean of H_noisy_in_test:", ((H_noisy_in_test)**2).mean())
print("Mean of H_true_out_test:", ((H_true_out_test)**2).mean())
print("Mean of decoded_channel:", ((decoded_channel)**2).mean())

nmse1=zeros((data_num_test,1), dtype=float)
nmse2=zeros((data_num_test,1), dtype=float)

for n in range(data_num_test):
    MSE1 = ((H_true_out_test[n,:,:,:] - H_noisy_in_test[n,:,:,:]) ** 2).sum()
    MSE2=((H_true_out_test[n,:,:,:]-decoded_channel[n,:,:,:])**2).sum()
    norm_real=((H_true_out_test[n,:,:,:])**2).sum()
    nmse1[n] = MSE1 / norm_real
    nmse2[n]=MSE2/norm_real
print(nmse1.sum()/(data_num_test), nmse2.sum()/(data_num_test))
