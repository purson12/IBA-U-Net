from __future__ import division
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.optimizers import *
from keras.layers import *

def RI(inputs, filter_size1, filter_size2, filter_size3, filter_size4):
    cnn1 = Conv2D(filter_size1, (3, 3), padding='same', activation="relu")(inputs)
    cnn2 = Conv2D(filter_size2, (3, 3), padding='same', activation="relu")(cnn1)
    cnn3 = Conv2D(filter_size3, (3, 3), padding='same', activation="relu")(cnn2)
    cnn = Conv2D(filter_size4, (1, 1), padding='same', activation="relu")(inputs)
    concat = Concatenate()([cnn1, cnn2, cnn3])
    add = Add()([concat, cnn])
    return add

def res_path(inputs, filter_size):
    cnn1 = Conv2D(filter_size, (1, 1), padding='same', activation="relu")(inputs)
    cnn2 = Conv2D(filter_size, (1, 1), padding='same', activation="relu")(inputs)
    add = concatenate([cnn1, cnn2], axis=3)
    return add



def Attentive_BConvLSTM(down_layer, layer, data_format='channels_last'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]
    print(down_layer.shape)
    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)
    print(up.shape)
    layer = Attentive_BconvLSTM_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def Attentive_BconvLSTM_2d(x, g, inter_channel, data_format='channels_last'):
    print(x.shape)
    print(g.shape)

    g = Conv2D(inter_channel*2, [1, 1], strides=[1, 1], data_format=data_format)(g)
    # f(?,g_height,g_width,inter_channel)
    x = Reshape(target_shape=(1, x.get_shape().as_list()[1], x.get_shape().as_list()[2], inter_channel*2))(x)
    g = Reshape(target_shape=(1, g.get_shape().as_list()[1], g.get_shape().as_list()[2], inter_channel*2))(x)
    merge = concatenate([x, g], axis=1)

    f = ConvLSTM2D(filters=inter_channel, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge)
    f = Activation('relu')(f)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)
    rate = Activation('sigmoid')(psi_f)

    x = multiply([x, rate])
    att_x = Reshape(target_shape=(x.get_shape().as_list()[2], x.get_shape().as_list()[3], x.get_shape().as_list()[4]))(x)
    return att_x

#Proposed IBA-U-net
def IBA_unet(input_size=(256, 256, 1), data_format='channels_last'):
    inputs = Input(input_size)
    N = input_size[0]
    skips = []
    new_inception1 = RI(inputs, 11, 21, 32, 64)#（N, N, 64)

    skips.append(new_inception1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(new_inception1)

    new_inception2 = RI(pool1, 21, 43, 64, 128)#(N/2, N/2, 128)
    skips.append(new_inception2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(new_inception2)

    new_inception3 = RI(pool2, 43, 85, 128, 256)#(N/4, N/4, 256)
    skips.append(new_inception3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(new_inception3)

    new_inception4 = RI(pool3, 85, 171, 256, 512)#(N/8, N/8, 512)
    skips.append(new_inception4)
    drop4 = Dropout(0.5)(new_inception4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    x = RI(pool4, 171, 341, 512, 1024)

    x = Dropout(0.5)(x)

    x = Attentive_BConvLSTM(x, skips[3], data_format=data_format)
    x = RI(x, 85, 171, 256, 512)

    x = Attentive_BConvLSTM(x, skips[2], data_format=data_format)
    x = RI(x, 43, 85, 128, 256)

    x = Attentive_BConvLSTM(x, skips[1], data_format=data_format)
    x = RI(x, 21, 43, 64, 128)

    x = Attentive_BConvLSTM(x, skips[0], data_format=data_format)
    x = RI(x, 11, 21, 32, 64)
    sigmoid = Conv2D(1, (1, 1), padding='same', activation="sigmoid")(x)

    model = Model(inputs, sigmoid)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model