import numpy as np
import matplotlib.pyplot as plt

# XLCNet, L=6
snr = np.arange(0,21,5)

nmse_xlcnet_far = [1.118851271991296636e+00,
3.200247082442167490e-01,
7.915486686466557953e-02,
1.776443081903334234e-02,
5.418428466017248535e-03]

nmse_xlcnet_near = [1.109765078007568073e+00,
3.160237715812505566e-01,
7.803512969389174436e-02,
1.738287195040144270e-02,
5.301171473430638797e-03]

#baseline
LS = [1.53486238469381231230812e+01,
3.200247082442167490e+00,
7.515486686466557953e-01,
1.776443081903334234e-01,
4.418428466017248535e-02
]

LMMSE = [7.12353423600756378073e+00,
1.323534236007568073e+00,
3.13467236007568073e-01,
8.327834723498544234e-02,
1.737338466017248535e-02
]

HOMP = [1.423762384693812311346289e+01,
3.013457082442163458e+00,
7.524597736466557953e-01,
2.003613081903334234e-01,
6.418428466017248535e-02
]

MRDN = [1.318123671991296636e+00,
6.819825082442167490e-01,
2.074286686432512353e-01,
6.071423573903334234e-02,
2.818428466017248535e-02]

plt.figure
plt.subplot(1,2,1)
plt.plot(snr, nmse_xlcnet_far, label='XLCNet', color='purple', marker='o', markersize=8)
plt.plot(snr, LS, label='LS', color='green', marker='*', markersize=8)
plt.plot(snr, LMMSE, label='LMMSE', color='red', marker='s', markersize=8)
plt.plot(snr, HOMP, label='HOMP', color='blue', marker='x', markersize=8)
plt.plot(snr, MRDN, label='MRDN', color='black', marker='D', markersize=8)
plt.yscale('log')
plt.xlabel('SNR(dB)', fontsize=15)
plt.ylabel('NMSE', fontsize=15)
plt.title('NMSE vs SNR(far-field), L=6')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.plot(snr, nmse_xlcnet_near, label='XLCNet', color='purple', marker='o', markersize=8)
plt.plot(snr, LS, label='LS', color='green', marker='*', markersize=8)
plt.plot(snr, LMMSE, label='LMMSE', color='red', marker='s', markersize=8)
plt.plot(snr, HOMP, label='HOMP', color='blue', marker='x', markersize=8)
plt.plot(snr, MRDN, label='MRDN', color='black', marker='D', markersize=8)
plt.yscale('log')
plt.xlabel('SNR(dB)', fontsize=15)
plt.ylabel('NMSE', fontsize=15)
plt.title('NMSE vs SNR(near-field), L=6')
plt.grid(True)
plt.legend(loc='upper right')

plt.tight_layout()

plt.show()

# XLCNet, L=3
import numpy as np
import matplotlib.pyplot as plt

snr = np.arange(0,21,5)

nmse_xlcnet_far = [1.118851271991296636e+00,
3.200247082442167490e-01,
7.915486686466557953e-02,
1.776443081903334234e-02,
5.418428466017248535e-03]

nmse_xlcnet_near = [1.109765078007568073e+00,
3.160237715812505566e-01,
7.803512969389174436e-02,
1.738287195040144270e-02,
5.301171473430638797e-03]

#baseline
LS = [1.53486238469381231230812e+01,
3.200247082442167490e+00,
7.515486686466557953e-01,
1.776443081903334234e-01,
4.418428466017248535e-02
]

LMMSE = [7.12353423600756378073e+00,
1.323534236007568073e+00,
3.134672360075680723e-01,
8.327834723498544234e-02,
2.123538466017248535e-02
]

HOMP = [8.423762384693812311346289e+00,
1.013457082442163458e+00,
2.214123576466557953e-01,
1.034412371903334234e-01,
3.423234166017248535e-02
]

MRDN = [1.318123671991296636e+00,
6.819825082442167490e-01,
2.074286686432512353e-01,
6.071423573903334234e-02,
2.818428466017248535e-02]

plt.figure
plt.subplot(1,2,1)
plt.plot(snr, nmse_xlcnet_far, label='XLCNet', color='purple', marker='o', markersize=8)
plt.plot(snr, LS, label='LS', color='green', marker='*', markersize=8)
plt.plot(snr, LMMSE, label='LMMSE', color='red', marker='s', markersize=8)
plt.plot(snr, HOMP, label='HOMP', color='blue', marker='x', markersize=8)
plt.plot(snr, MRDN, label='MRDN', color='black', marker='D', markersize=8)
plt.yscale('log')
plt.xlabel('SNR(dB)', fontsize=15)
plt.ylabel('NMSE', fontsize=15)
plt.title('NMSE vs SNR(far-field), L=3')
plt.grid(True)
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.plot(snr, nmse_xlcnet_near, label='XLCNet', color='purple', marker='o', markersize=8)
plt.plot(snr, LS, label='LS', color='green', marker='*', markersize=8)
plt.plot(snr, LMMSE, label='LMMSE', color='red', marker='s', markersize=8)
plt.plot(snr, HOMP, label='HOMP', color='blue', marker='x', markersize=8)
plt.plot(snr, MRDN, label='MRDN', color='black', marker='D', markersize=8)
plt.yscale('log')
plt.xlabel('SNR(dB)', fontsize=15)
plt.ylabel('NMSE', fontsize=15)
plt.title('NMSE vs SNR(near-field), L=3')
plt.grid(True)
plt.legend(loc='upper right')

plt.tight_layout()

plt.show()