import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import cv2
import seaborn
seaborn.set_style('darkgrid')
import shutil
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization,ReLU
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import time
from sklearn.metrics import f1_score
import scipy.stats as stats
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

def gabor_kernel(frequency, sigma_x, sigma_y, theta=0, offset=0, ks=61):
    w = ks // 2
    grid_val = tf.range(-w, w+1, dtype=tf.float32)
    x, y = tf.meshgrid(grid_val, grid_val)
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)
    g = tf.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= tf.cos(2 * np.pi * frequency * rotx + offset)

    return g


class Identity(layers.Layer):
    def call(self, inputs):
        return inputs


class GFB(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weight = tf.Variable(tf.zeros(
            (kernel_size, kernel_size, in_channels, out_channels)), trainable=False)

    def call(self, inputs):
        return tf.nn.conv2d(inputs, self.weight, strides=[1, self.stride, self.stride, 1], padding='SAME')

    def initialize(self, sf, theta, sigx, sigy, phase):
        random_channel = tf.random.uniform(
            (self.out_channels,), minval=0, maxval=self.in_channels, dtype=tf.int32)
#         for i in range(self.out_channels):
#             self.weight[i, random_channel[i]] = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
#                                                              theta=theta[i], offset=phase[i], ks=self.kernel_size)
#         self.weight.assign(self.weight)
        for i in range(self.out_channels):
            gabor = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                     theta=theta[i], offset=phase[i], ks=self.kernel_size)

            # 使用scatter_nd_update更新权重
#             tensor_i = tf.constant(i, dtype=tf.int32)  # 将 i 转换为张量
#             indices = tf.constant([[tensor_i, random_channel[i]]], dtype=tf.int32)
# #             indices = tf.constant([[i, random_channel[i]]], dtype=tf.int32)
#             updates = tf.expand_dims(gabor, axis=0)
#             self.weight = tf.tensor_scatter_nd_update(self.weight, indices, updates)
#             tensor_i = tf.constant(i, dtype=tf.int32)  # 将 i 转换为张量
#             tensor_channel = tf.constant(random_channel[i], dtype=tf.int32)  # 将 random_channel[i] 转换为张量
#             indices = tf.expand_dims(tf.stack([tensor_i, tensor_channel]), axis=0)  # 堆叠张量并增加维度
#             updates = tf.expand_dims(gabor, axis=0)
#             self.weight = tf.tensor_scatter_nd_update(self.weight, indices, updates)
            kernel = gabor_kernel(frequency=sf[i], sigma_x=sigx[i], sigma_y=sigy[i],
                                  theta=theta[i], offset=phase[i], ks=self.kernel_size)
            self.weight[:, :, random_channel[i], i].assign(kernel)

class VOneBlock(layers.Layer):
    def __init__(self, sf, theta, sigx, sigy, phase,
                 k_exc=25, noise_mode=None, noise_scale=1, noise_level=1,
                 simple_channels=128, complex_channels=128, ksize=25, stride=4, input_size=224):
        super().__init__()

        self.in_channels = 3
        self.simple_channels = simple_channels
        self.complex_channels = complex_channels
        self.out_channels = simple_channels + complex_channels
        self.stride = stride
        self.input_size = input_size

        self.sf = sf
        self.theta = theta
        self.sigx = sigx
        self.sigy = sigy
        self.phase = phase
        self.k_exc = k_exc

        self.set_noise_mode(noise_mode, noise_scale, noise_level)
        self.fixed_noise = None

        self.simple_conv_q0 = GFB(
            self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q1 = GFB(
            self.in_channels, self.out_channels, ksize, stride)
        self.simple_conv_q0.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase)
        self.simple_conv_q1.initialize(sf=self.sf, theta=self.theta, sigx=self.sigx, sigy=self.sigy,
                                       phase=self.phase + np.pi / 2)

        self.simple = layers.ReLU()
        self.complex = Identity()
        self.gabors = Identity()
        self.noise = layers.ReLU()
        self.output_layer = Identity()

    def call(self, inputs):
        x = self.gabors_f(inputs)
        x = self.noise_f(x)
        x = self.output_layer(x)
        return x

    def gabors_f(self, inputs):
        s_q0 = self.simple_conv_q0(inputs)
        s_q1 = self.simple_conv_q1(inputs)
        c = self.complex(tf.math.sqrt(s_q0[:, :, :, self.simple_channels:] ** 2 +
                                      s_q1[:, :, :, self.simple_channels:] ** 2) / np.sqrt(2))
        s = self.simple(s_q0[:, :, :, :self.simple_channels])

        return self.gabors(self.k_exc * tf.concat((s, c), axis=-1))

    def noise_f(self, inputs):
        x = inputs
        if self.noise_mode == 'neuronal':
            eps = 10e-5
            x *= self.noise_scale
            x += self.noise_level
            if self.fixed_noise is not None:
                x += self.fixed_noise * tf.math.sqrt(tf.nn.relu(x) + eps)
            else:
                x += tf.random.normal(tf.shape(x), stddev=1) * \
                    tf.math.sqrt(tf.nn.relu(x) + eps)
            x -= self.noise_level
            x /= self.noise_scale
        if self.noise_mode == 'gaussian':
            if self.fixed_noise is not None:
                x += self.fixed_noise * self.noise_scale
            else:
                x += tf.random.normal(tf.shape(x), stddev=1) * self.noise_scale
        return self.noise(x)

    def set_noise_mode(self, noise_mode=None, noise_scale=1, noise_level=1):
        self.noise_mode = noise_mode
        self.noise_scale = noise_scale
        self.noise_level = noise_level

    def fix_noise(self, batch_size=256, seed=None):
        if seed:
            tf.random.set_seed(seed)
        noise_mean = tf.zeros((batch_size, int(
            self.input_size / self.stride), int(self.input_size / self.stride), self.out_channels))
        if self.noise_mode:
            self.fixed_noise = tf.random.normal(noise_mean.shape, stddev=1)

    def unfix_noise(self):
        self.fixed_noise = None
        
    
def sample_dist(hist, bins, ns, scale='linear'):
    rand_sample = np.random.rand(ns)
    if scale == 'linear':
        rand_sample = np.interp(
            rand_sample, np.hstack(([0], hist.cumsum())), bins)
    elif scale == 'log2':
        rand_sample = np.interp(rand_sample, np.hstack(
            ([0], hist.cumsum())), np.log2(bins))
        rand_sample = 2**rand_sample
    elif scale == 'log10':
        rand_sample = np.interp(rand_sample, np.hstack(
            ([0], hist.cumsum())), np.log10(bins))
        rand_sample = 10**rand_sample
    return rand_sample


def generate_gabor_param(features, seed=0, rand_flag=False, sf_corr=0, sf_max=9, sf_min=0):
    # Generates random sample
    np.random.seed(seed)

    phase_bins = np.array([0, 360])
    phase_dist = np.array([1])

    if rand_flag:
        print('Uniform gabor parameters')
        ori_bins = np.array([0, 180])
        ori_dist = np.array([1])

        nx_bins = np.array([0.1, 10**0.2])
        nx_dist = np.array([1])

        ny_bins = np.array([0.1, 10**0.2])
        ny_dist = np.array([1])

        # sf_bins = np.array([0.5, 8])
        # sf_dist = np.array([1])

        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8])
        sf_dist = np.array([1,  1,  1, 1, 1, 1, 1, 1])

        sfmax_ind = np.where(sf_bins < sf_max)[0][-1]
        sfmin_ind = np.where(sf_bins >= sf_min)[0][0]

        sf_bins = sf_bins[sfmin_ind:sfmax_ind+1]
        sf_dist = sf_dist[sfmin_ind:sfmax_ind]

        sf_dist = sf_dist / sf_dist.sum()
    else:
        print('Neuronal distributions gabor parameters')
        # DeValois 1982a
        ori_bins = np.array([-22.5, 22.5, 67.5, 112.5, 157.5])
        ori_dist = np.array([66, 49, 77, 54])
        ori_dist = ori_dist / ori_dist.sum()

        # Schiller 1976
        cov_mat = np.array([[1, sf_corr], [sf_corr, 1]])

        # Ringach 2002b
        nx_bins = np.logspace(-1, 0.2, 6, base=10)
        ny_bins = np.logspace(-1, 0.2, 6, base=10)
        n_joint_dist = np.array([[2.,  0.,  1.,  0.,  0.],
                                 [8.,  9.,  4.,  1.,  0.],
                                 [1.,  2., 19., 17.,  3.],
                                 [0.,  0.,  1.,  7.,  4.],
                                 [0.,  0.,  0.,  0.,  0.]])
        n_joint_dist = n_joint_dist / n_joint_dist.sum()
        nx_dist = n_joint_dist.sum(axis=1)
        nx_dist = nx_dist / nx_dist.sum()
        ny_dist_marg = n_joint_dist / n_joint_dist.sum(axis=1, keepdims=True)

        # DeValois 1982b
        sf_bins = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8])
        sf_dist = np.array([4,  4,  8, 25, 32, 26, 28, 12])

        sfmax_ind = np.where(sf_bins <= sf_max)[0][-1]
        sfmin_ind = np.where(sf_bins >= sf_min)[0][0]

        sf_bins = sf_bins[sfmin_ind:sfmax_ind+1]
        sf_dist = sf_dist[sfmin_ind:sfmax_ind]

        sf_dist = sf_dist / sf_dist.sum()

    phase = sample_dist(phase_dist, phase_bins, features)
    ori = sample_dist(ori_dist, ori_bins, features)
    ori[ori < 0] = ori[ori < 0] + 180

    if rand_flag:
        sf = sample_dist(sf_dist, sf_bins, features, scale='log2')
        nx = sample_dist(nx_dist, nx_bins, features, scale='log10')
        ny = sample_dist(ny_dist, ny_bins, features, scale='log10')
    else:

        samps = np.random.multivariate_normal([0, 0], cov_mat, features)
        samps_cdf = stats.norm.cdf(samps)

        nx = np.interp(samps_cdf[:, 0], np.hstack(
            ([0], nx_dist.cumsum())), np.log10(nx_bins))
        nx = 10**nx

        ny_samp = np.random.rand(features)
        ny = np.zeros(features)
        for samp_ind, nx_samp in enumerate(nx):
            bin_id = np.argwhere(nx_bins < nx_samp)[-1]
            ny[samp_ind] = np.interp(ny_samp[samp_ind], np.hstack(([0], ny_dist_marg[bin_id, :].cumsum())),
                                     np.log10(ny_bins))
        ny = 10**ny

        sf = np.interp(samps_cdf[:, 1], np.hstack(
            ([0], sf_dist.cumsum())), np.log2(sf_bins))
        sf = 2**sf

    return sf, ori, phase, nx, ny


def VOneNet(sf_corr=0.75, sf_max=9, sf_min=0, rand_param=False, gabor_seed=0,
            simple_channels=256, complex_channels=256,
            noise_mode='neuronal', noise_scale=0.35, noise_level=0.07, k_exc=25,
            model_arch='resnet50', image_size=300, visual_degrees=8, ksize=25, stride=4, class_count=14):

    out_channels = simple_channels + complex_channels

    sf, theta, phase, nx, ny = generate_gabor_param(
        out_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min)

    gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
                    'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                    'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
    arch_params = {'k_exc': k_exc, 'arch': model_arch,
                   'ksize': ksize, 'stride': stride}

    # Conversions
    ppd = image_size / visual_degrees

    sf = sf / ppd
    sigx = nx / sf
    sigy = ny / sf
    theta = theta/180 * np.pi
    phase = phase / 180 * np.pi

    vone_block = VOneBlock(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase,
                           k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                           simple_channels=simple_channels, complex_channels=complex_channels,
                           ksize=ksize, stride=stride, input_size=image_size)
    
    
    
    
    img_shape=(image_size,image_size,3)
#     model_back_end=tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet",input_shape=img_shape, pooling='max')
#     inputs = tf.keras.Input(shape=(image_size, image_size, 3))
#     x = vone_block(inputs)
#     if model_arch:
    bottleneck = Conv2D(3, kernel_size=1, strides=1, use_bias=False,kernel_initializer=tf.keras.initializers.HeNormal())
#         # if model_arch.lower() == 'resnet50':
#         #     print('Model: ', 'VOneResnet50')
#         #     model_back_end = ResNetBackEnd(
#         #         block=Bottleneck, layers=[3, 4, 6, 3])
#         # elif model_arch.lower() == 'alexnet':
#         #     print('Model: ', 'VOneAlexNet')
#         #     model_back_end = AlexNetBackEnd()
#         # elif model_arch.lower() == 'cornets':
#         #     print('Model: ', 'VOneCORnet-S')
#         #     model_back_end = CORnetSBackEnd()
#         # TODO: change model_back_end to our block
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = vone_block(inputs)
    x = bottleneck(x)
    
    model_back_end=tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet",input_shape=x.shape[1:], pooling='max')
    model_back_end.trainable = True
#     print(backend_input.shape)

    
    # 用 bottleneck 的输出替换模型后端的输入张量
    backend_output = model_back_end(x)
    
    
    x=backend_output
    x=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 )(x)
    x=Dense(1024,activation='relu')(x)
    outputs=Dense(class_count, activation='sigmoid')(x)

    
#     outputs = model_back_end.output
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(Adamax(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy']) 
#     else:
#         print('Model: ', 'VOneNet')
#         model = vone_block

    model.image_size = image_size
    model.visual_degrees = visual_degrees
    model.gabor_parms = gabor_params
    model.arch_params = arch_params
    return model



