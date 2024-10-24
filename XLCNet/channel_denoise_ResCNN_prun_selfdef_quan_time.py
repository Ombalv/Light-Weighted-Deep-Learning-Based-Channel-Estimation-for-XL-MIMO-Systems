import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, Conv2D, BatchNormalization, MaxPool2D, Add, Activation, Subtract, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from numpy import *
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #allow growth
import scipy.io as sio
import tempfile
import time
import h5py

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
data1 = sio.loadmat('...\\Channel_f0n3_256ANTS_10by200')
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

loop_num = 10
time_origin=0
time_prun=0
time_quan=0
for i in range(loop_num):
    # load model
    ResCNN = load_model('ResCNN9_f1n5_256ANTS_1Kby100data_10dB_200ep.hdf5')
    t1 = time.time()
    decoded_channel = ResCNN.predict(H_noisy_in_test)
    t2 = time.time()
    time_origin += t2-t1
    nmse1=zeros((data_num_test,1), dtype=float)
    nmse2=zeros((data_num_test,1), dtype=float)
    for n in range(data_num_test):
        MSE1 = ((H_true_out_test[n,:,:,:] - H_noisy_in_test[n,:,:,:]) ** 2).sum()
        MSE2=((H_true_out_test[n,:,:,:]-decoded_channel[n,:,:,:])**2).sum()
        norm_real=((H_true_out_test[n,:,:,:])**2).sum()
        nmse1[n] = MSE1 / norm_real
        nmse2[n]=MSE2/norm_real


    ############## Pruning ##################
    PrunResCNN = load_model('ResCNN9_f1n5_256ANTS_1Kby100data_10dB_200ep_Prun09_50ep.hdf5')
    t_prun1 = time.time()
    decoded_channel = PrunResCNN.predict(H_noisy_in_test)
    t_prun2 = time.time()
    time_prun += t_prun2 - t_prun1
    nmse1=zeros((data_num_test,1), dtype=float)
    nmse2=zeros((data_num_test,1), dtype=float)
    for n in range(data_num_test):
        MSE1 = ((H_true_out_test[n,:,:,:] - H_noisy_in_test[n,:,:,:]) ** 2).sum()
        MSE2=((H_true_out_test[n,:,:,:]-decoded_channel[n,:,:,:])**2).sum()
        norm_real=((H_true_out_test[n,:,:,:])**2).sum()
        nmse1[n] = MSE1 / norm_real
        nmse2[n]=MSE2/norm_real

    ## Build DNN model
    K = 3
    input_dim = (Nx, Ny, 2)
    output_dim = 2
    inp = Input(shape=input_dim)
    xn = Conv2D(filters=64, kernel_size=(K, K), padding='Same', activation='relu')(inp)
    xn = BatchNormalization()(xn)
    xn = Conv2D(filters=64, kernel_size=(K, K), padding='Same', activation='relu')(xn)
    xn = BatchNormalization()(xn)
    xn = Conv2D(filters=64, kernel_size=(K, K), padding='Same', activation='relu')(xn)
    xn = BatchNormalization()(xn)
    xn = Conv2D(filters=64, kernel_size=(K, K), padding='Same', activation='relu')(xn)
    xn = BatchNormalization()(xn)
    xn = Conv2D(filters=64, kernel_size=(K, K), padding='Same', activation='relu')(xn)
    xn = BatchNormalization()(xn)
    xn = Conv2D(filters=64, kernel_size=(K, K), padding='Same', activation='relu')(xn)
    xn = BatchNormalization()(xn)
    xn = Conv2D(filters=64, kernel_size=(K, K), padding='Same', activation='relu')(xn)
    xn = BatchNormalization()(xn)
    xn = Conv2D(filters=64, kernel_size=(K, K), padding='Same', activation='relu')(xn)
    xn = BatchNormalization()(xn)
    xn = Conv2D(filters=output_dim, kernel_size=(K, K), padding='Same', activation='linear')(xn)
    x1 = Subtract()([inp, xn])

    model1 = Model(inputs=inp, outputs=x1)

    adam = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model1.compile(optimizer=adam, loss='mse')


    # quantizing weights
    bit_array = np.arange(7,9,1)
    for b in range(len(bit_array)):
        bits = bit_array[b]  #quantization bits

        unquan_weights_total = PrunResCNN.get_weights()
        i = 0
        unquan_weights = []
        for layer in PrunResCNN.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                unquan_weights.append(layer.get_weights())
                i = i + 1
        # print(i)
        quan_weights = unquan_weights

        for l in range(i):
            weights_kernel = quan_weights[l][0]
            weights_bias = quan_weights[l][1]

            weights_kernel_vec = np.reshape(weights_kernel, (np.size(weights_kernel),))
            weights_kernel_nozero = np.delete(weights_kernel_vec, np.argwhere(weights_kernel_vec==0))
            weights_kernel_max = np.max(weights_kernel_nozero)
            weights_kernel_min = np.min(weights_kernel_nozero)

            #for entry in enumerate(weights_kernel):
            nonzero_index = np.transpose(np.array(np.nonzero(weights_kernel)))
            len_index = len(nonzero_index)
            #dim = np.array(weights_kernel.shape)
            #len_dim = len(dim)
            for m in range(len_index):
                #nonzero_index_vec = dim[1] * dim[2] * dim[3] * nonzero_index[m][0] + dim[2] * dim[3] * nonzero_index[m][1] + dim[3] * nonzero_index[m][2] + nonzero_index[m][3]
                weights_kernel[nonzero_index[m][0],nonzero_index[m][1],nonzero_index[m][2],nonzero_index[m][3]] = np.round(
                    (weights_kernel[nonzero_index[m][0],nonzero_index[m][1],nonzero_index[m][2],nonzero_index[m][3]]-weights_kernel_min)/(weights_kernel_max-weights_kernel_min)*(2**bits-1))*(weights_kernel_max-weights_kernel_min)/(2**bits-1)+weights_kernel_min

            weights_bias_vec = np.reshape(weights_bias, (np.size(weights_bias),))
            weights_bias_nozero = np.delete(weights_bias_vec, np.argwhere(weights_bias_vec == 0))
            weights_bias_max = np.max(weights_bias_nozero)
            weights_bias_min = np.min(weights_bias_nozero)

            #for n, entry in enumerate(weights_bias):
            nonzero_index = np.transpose(np.array(np.nonzero(weights_bias)))
            len_index = len(nonzero_index)
            for m in range(len_index):
                weights_bias[nonzero_index[m][0]] = np.round(
                    (weights_bias[nonzero_index[m][0]] - weights_bias_min) / (weights_bias_max - weights_bias_min) * (2 ** bits - 1))*(weights_bias_max-weights_bias_min)/(2**bits-1)+weights_bias_min

            quan_weights[l][0] = weights_kernel
            quan_weights[l][1] = weights_bias

        model1.set_weights(unquan_weights_total)

        j = 0
        for layer in model1.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer.set_weights(quan_weights[j])
                j = j + 1
        #print(j)

        t_quan1 = time.time()
        decoded_channel = model1.predict(H_noisy_in_test)
        t_quan2 = time.time()
        if bits == 8:
            time_quan += t_quan2 - t_quan1
        nmse1=zeros((data_num_test,1), dtype=float)
        nmse2=zeros((data_num_test,1), dtype=float)
        for n in range(data_num_test):
            MSE1 = ((H_true_out_test[n,:,:,:] - H_noisy_in_test[n,:,:,:]) ** 2).sum()
            MSE2=((H_true_out_test[n,:,:,:]-decoded_channel[n,:,:,:])**2).sum()
            norm_real=((H_true_out_test[n,:,:,:])**2).sum()
            nmse1[n] = MSE1 / norm_real
            nmse2[n]=MSE2/norm_real

print('Original time:', time_origin/loop_num/data_num_test)
print('Prune time:', time_prun/loop_num/data_num_test)
print('Quantization time:', time_quan/loop_num/data_num_test)

