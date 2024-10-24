# Light-Weighted-Deep-Learning-Based-Channel-Estimation-for-XL-MIMO-Systems
This repo contains two different neural networks for solving the tricky channel estimation problem for the XL-MIMO system.

## XL-MIMO Channel Network (XLCNet)
### model
In dir "/model", there are several well-performed models selected from multiple experiments with differences in the number of antennas, data size, noise intensity, learning rate, and pruning scale.

### data
In dir "/data", the input data matrix to transmit files are put in this dir, training models with near-field and far-field channel matrix separately, and evaluating the performance of our trained models with matrix mixed with far-field and near-field data.

### ChannelData_for_ResCNN.m
This is the matlab code to generate the pseudo channel matrix for training and evaluating.

### Other .py files
Original channel network, pruned channel network, quantized channel network and their corresponding testing code

## Polar-domian Multi-Scale Residual Dense Network (P-MSRDN)
Quite a straightforward code structure,
Just click in and you will understand.
