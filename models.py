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


def BConvLSTM_Unet(input_size=(256, 256, 1)):
    N = input_size[0]
    inputs = Input(input_size)
    conv1 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
#(N, N, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#（N/2 ,
    conv2 = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # D1
    conv4 = Conv2D(512, 5, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    drop4_1 = Dropout(0.5)(conv4)
    # D2
    conv4_2 = Conv2D(512, 5, activation='relu', padding='same', kernel_initializer='he_normal')(drop4_1)
    conv4_2 = Dropout(0.5)(conv4_2)
    # D3
    merge_dense = concatenate([conv4_2, drop4_1], axis=3)
    conv4_3 = Conv2D(512, 5, activation='relu', padding='same', kernel_initializer='he_normal')(merge_dense)
    drop4_3 = Dropout(0.5)(conv4_3)

    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(drop4_3)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Activation('relu')(up6)

    x1 = Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(drop3)
    x2 = Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(up6)
    merge6 = concatenate([x1, x2], axis=1)
    merge6 = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge6)

    conv6 = Conv2D(256, 5, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv6)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Activation('relu')(up7)
    print(conv2.shape)
    x1 = Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(conv2)
    print(x1.shape)
    x2 = Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(up7)
    print(x2.shape)
    merge7 = concatenate([x1, x2], axis=1)
    print(merge7.shape)
    merge7 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge7)
    print(merge7.shape)
    conv7 = Conv2D(128, 5, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer='he_normal')(conv7)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Activation('relu')(up8)

    x1 = Reshape(target_shape=(1, N, N, 64))(conv1)
    x2 = Reshape(target_shape=(1, N, N, 64))(up8)
    merge8 = concatenate([x1, x2], axis=1)
    merge8 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(merge8)

    conv8 = Conv2D(64, 5, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv9 = Conv2D(1, 1, activation='sigmoid')(conv8)

    model = Model(input=inputs, output=conv9)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def RI_unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    unet_block1 = RI(inputs, 11, 21, 32, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(unet_block1)

    unet_block2 = RI(pool1, 21, 43, 64, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(unet_block2)

    unet_block3 = RI(pool2, 43, 85, 128, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(unet_block3)

    unet_block4 = RI(pool3, 85, 171, 256, 512)
    drop4 = Dropout(0.5)(unet_block4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    unet_block5 = RI(pool4, 171, 341, 512, 1024)

    drop5 = Dropout(0.5)(unet_block5)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)

    unet_block6 = RI(merge6, 85, 171, 256, 512)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(unet_block6))
    merge7 = concatenate([unet_block3, up7], axis=3)

    unet_block7 = RI(merge7, 43, 85, 128, 256)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(unet_block7))
    merge8 = concatenate([unet_block2, up8], axis=3)

    unet_block8 = RI(merge8, 21, 43, 64, 128)  # (64,64,256)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(unet_block8))
    merge9 = concatenate([unet_block1, up9], axis=3)

    unet_block9 = RI(merge9, 11, 21, 32, 64)  # (64,64,256)


    unet_block9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(unet_block9)
    unet_block10 = Conv2D(1, 1, activation='sigmoid')(unet_block9)

    model = Model(input=inputs, output=unet_block10)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model



def block(prevlayer, a, b, pooling):
    conva = Conv2D(a, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(prevlayer)
    conva = BatchNormalization()(conva)
    conva = Conv2D(b, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(conva)
    conva = BatchNormalization()(conva)
    if True == pooling:
        conva = MaxPooling2D(pool_size=(2, 2))(conva)

    convb = Conv2D(a, 5, activation='relu', padding='same', kernel_initializer = 'he_normal')(prevlayer)
    convb = BatchNormalization()(convb)
    convb = Conv2D(b, 5, activation='relu', padding='same', kernel_initializer = 'he_normal')(convb)
    convb = BatchNormalization()(convb)
    if True == pooling:
        convb = MaxPooling2D(pool_size=(2, 2))(convb)

    convc = Conv2D(b, 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(prevlayer)
    convc = BatchNormalization()(convc)
    if True == pooling:
        convc = MaxPooling2D(pool_size=(2, 2))(convc)

    convd = Conv2D(a, 3, activation='relu', padding='same', kernel_initializer = 'he_normal')(prevlayer)
    convd = BatchNormalization()(convd)
    convd = Conv2D(b, 1, activation='relu', padding='same', kernel_initializer = 'he_normal')(convd)
    convd = BatchNormalization()(convd)
    if True == pooling:
        convd = MaxPooling2D(pool_size=(2, 2))(convd)

    up = concatenate([conva, convb, convc, convd])
    return up


def inception_unet(input_size=(256, 256, 1), data_format='channels_last'):
    inputs = Input(input_size)

    conv1 = block(inputs, 8, 16, True)

    conv2 = block(conv1, 16, 32, True)

    conv3 = block(conv2, 32, 64, True)

    conv4 = block(conv3, 64, 128, True)

    conv5 = block(conv4, 128, 256, True)

    # **** decoding ****
    xx = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    up1 = block(xx, 256, 64, False)

    xx = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(up1), conv3], axis=3)
    up2 = block(xx, 128, 32, False)

    xx = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(up2), conv2], axis=3)
    up3 = block(xx, 64, 16, False)

    xx = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(up3), conv1], axis=3)
    up4 = block(xx, 32, 8, False)

    xx = concatenate([Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(up4), inputs], axis=3)

    xx = Conv2D(16, (3, 3), activation='relu', padding='same')(xx)
    #    xx = concatenate([xx, conv1a])

    xx = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(xx)

    model = Model(inputs=[inputs], outputs=[xx])
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model



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


def RI_BConvLSTM_unet(pretrained_weights=None, input_size=(256, 256, 1), lr=0.001):
    N = input_size[0]
    inputs = Input(input_size)  # （512,512,1）

    res_block1 = RI(inputs, 5, 11, 16, 32)  # (None, 512, 512, 32)
    pool1 = MaxPool2D()(res_block1)

    res_block2 = RI(pool1, 11, 21, 32, 64)# 256,256,64
    pool2 = MaxPool2D()(res_block2)

    res_block3 = RI(pool2, 21, 43, 64, 128)  # (128,128,128)
    pool3 = MaxPool2D()(res_block3)  # (64,64,128)

    res_block4 = RI(pool3, 43, 85, 128, 256)  # (64,64,256)
    pool4 = MaxPool2D()(res_block4)  # (32 , 32, 256)

    res_block5 = RI(pool4, 85, 171, 256, 512)  # (32,32,512)
    unsample = UpSampling2D()(res_block5)  # (64,64,512)

    # phi_g = Reshape(target_shape=(1, np.int32(N/8), np.int32(N/8), 128))(up5)
    res_path4 = res_path(res_block4, 256)  # (None, 64, 64, 512)
    x1 = Reshape(target_shape=(1, np.int32(N / 8), np.int32(N / 8), 512))(unsample)
    x2 = Reshape(target_shape=(1, np.int32(N / 8), np.int32(N / 8), 512))(res_path4)
    concat = concatenate([x1, x2], axis=1)
    concat = ConvLSTM2D(filters=256, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(concat)

    res_block6 = RI(concat, 43, 85, 128, 256)  # (None, 64, 64, 256)
    upsample = UpSampling2D()(res_block6)  # (None, 128, 128, 256)
    res_path3 = res_path(res_block3, 128)

    x1 = Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(upsample)
    x2 = Reshape(target_shape=(1, np.int32(N / 4), np.int32(N / 4), 256))(res_path3)
    concat = concatenate([x1, x2], axis=1)
    concat = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(concat)

    res_block7 = RI(concat, 21, 43, 64, 128)  # (None, 128, 128, 209)
    upsample = UpSampling2D()(res_block7)  # (None, 256, 256, 209)

    res_path2 = res_path(res_block2, 64)  # (256,256,64)

    x1 = Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(upsample)
    x2 = Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 128))(res_path2)
    concat = concatenate([x1, x2], axis=1)
    concat = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(concat)

    res_block8 = RI(concat, 11, 21, 32, 64)
    upsample = UpSampling2D()(res_block8)

    res_path1 = res_path(res_block1, 32)
    # res_path1 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same',kernel_initializer='he_normal')(res_path1)
    x1 = Reshape(target_shape=(1, np.int32(N), np.int32(N), 64))(upsample)
    x2 = Reshape(target_shape=(1, np.int32(N), np.int32(N), 64))(res_path1)
    concat = concatenate([x1, x2], axis=1)
    concat = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                        kernel_initializer='he_normal')(concat)

    res_block9 = RI(concat, 5, 11, 16, 32)

    x = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(res_block9)
    x = Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(input=inputs, output=x)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


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


def Attentive_Bconvlstm_unet(input_size=(256, 256, 1), data_format='channels_last'):
    inputs = Input(input_size)
    N = input_size[0]
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        features = features * 2
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = Attentive_BConvLSTM(x, skips[i], data_format=data_format)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    sigmoid = Conv2D(1, (1, 1), padding='same', activation="sigmoid")(x)

    model = Model(inputs, sigmoid)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

