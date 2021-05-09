# 未测试
# unet++
import numpy as np
import os
import tensorflow as tf
import skimage.io as io
import skimage.transform as trans
import numpy as np
from pandas import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.losses import *
from keras.metrics import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras import backend as keras
from keras.utils import plot_model
from metrics import *
from loss import *


# 块
def conv_bachnorm_relu_block(input_tensor, nb_filter, kernel_size=3):
    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding='same')(input_tensor)
    x = BatchNormalization(axis=2)(x)
    x = Activation('relu')(x)

    return x


# 模型结构
def NestedUNet(pretrained_weights=None, input_size=(256, 256, 1), using_deep_supervision=False):
    nb_filter = [32, 64, 128, 256, 512]
    inputs = Input(input_size)

    # conv1_1 代表第一层第一个，同理 conv2_1 代表第二层第一个
    # 第一层
    conv1_1 = conv_bachnorm_relu_block(inputs,nb_filter=nb_filter[0])
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(conv1_1)

    # 第二层
    conv2_1 = conv_bachnorm_relu_block(pool1, nb_filter=nb_filter[1])
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(conv2_1)

    # keras 自带上采样层
    # 融合块 1
    # up 1 次
    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=3)
    conv1_2 = conv_bachnorm_relu_block(conv1_2, nb_filter=nb_filter[0])

    # 第三层
    conv3_1 = conv_bachnorm_relu_block(pool2, nb_filter=nb_filter[2])
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(conv3_1)

    # up 1 次
    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=3)
    conv2_2 = conv_bachnorm_relu_block(conv2_2, nb_filter=nb_filter[1])

    # up 2 次
    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=3)
    conv1_3 = conv_bachnorm_relu_block(conv1_3, nb_filter=nb_filter[0])

    # 第四层
    conv4_1 = conv_bachnorm_relu_block(pool3, nb_filter=nb_filter[3])
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4')(conv4_1)

    # up 1 次
    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=3)
    conv3_2 = conv_bachnorm_relu_block(conv3_2, nb_filter=nb_filter[2])

    # up 2 次
    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=3)
    conv2_3 = conv_bachnorm_relu_block(conv2_3, nb_filter=nb_filter[1])

    # up 3 次
    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=3)
    conv1_4 = conv_bachnorm_relu_block(conv1_4, nb_filter=nb_filter[0])

    # 第五层
    conv5_1 = conv_bachnorm_relu_block(pool4, nb_filter=nb_filter[4])

    # up 1 次
    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=3)
    conv4_2 = conv_bachnorm_relu_block(conv4_2, nb_filter=nb_filter[3])

    # up 2 次
    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=3)
    conv3_3 = conv_bachnorm_relu_block(conv3_3, nb_filter=nb_filter[2])

    # up 3 次
    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=3)
    conv2_4 = conv_bachnorm_relu_block(conv2_4, nb_filter=nb_filter[1])

    # up 4 次
    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=3)
    conv1_5 = conv_bachnorm_relu_block(conv1_5, nb_filter=nb_filter[0])

    # deep_supervision
    nestnet_output_1 = Conv2D(1, (1, 1), activation='sigmoid', name='output_1', padding='same')(conv1_2)
    nestnet_output_2 = Conv2D(1, (1, 1), activation='sigmoid', name='output_2', padding='same')(conv1_3)
    nestnet_output_3 = Conv2D(1, (1, 1), activation='sigmoid', name='output_3', padding='same')(conv1_4)
    nestnet_output_4 = Conv2D(1, (1, 1), activation='sigmoid', name='output_4', padding='same')(conv1_5)

    if using_deep_supervision:
        model = Model(inputs=inputs, outputs=[nestnet_output_1,
                                              nestnet_output_2,
                                              nestnet_output_3,
                                              nestnet_output_4])
    else:
        model = Model(inputs=inputs, outputs=nestnet_output_4)


    # tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=Adam(lr=1e-4), loss=bce_dice_loss,
                  metrics=['accuracy', my_iou_metric])
    # optimizer:优化器及参数
    # loss:损失函数
    # metrics:评价指标

    # model.summary()
    plot_model(model, to_file='img/NestedUNet_model.png')
    # 加载预训练网络
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
