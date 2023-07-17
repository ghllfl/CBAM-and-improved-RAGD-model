
from __future__ import print_function
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import tensorflow as tf
import random
from tensorflow.keras.layers import Conv2D, Activation, Input, Dropout, MaxPooling2D, UpSampling2D, Conv2DTranspose, Concatenate, multiply, Add, LeakyReLU
from tensorflow.keras import regularizers
import tensorflow.keras.layers as layers


def expand(x):
    x = K.expand_dims(x, axis=-1)
    return x
def squeeze(x):
    x = K.squeeze(x, axis=-1)
    return x

def Res_BN_block(filter_num, input):
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    x = Conv2D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    # Residual/Skip connection
    res = Conv2D(filter_num, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    x = Add()([res, x])
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def Res_BN_block3d(filter_num, input):
    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(input)
    x = BatchNormalization()(x)
    x1 = Activation('relu')(x)
    x = Conv3D(filter_num, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = BatchNormalization()(x)
    # Residual/Skip connection
    res = Conv3D(filter_num, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    x = Add()([res, x])
    x = Activation('relu')(x)
    return x

def regularized_padded_conv(*args,**kwargs):
    randNum = random.randint(20000, 30000)
    layerName = 'Conv2d_spatial'+ str(randNum)
    return layers.Conv2D(*args,**kwargs,padding='same',name=layerName,kernel_regularizer=regularizers.l2(5e-5),
                         use_bias=False,kernel_initializer='glorot_normal')


#CBAM
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = regularized_padded_conv(1, kernel_size=kernel_size, strides=1, activation='sigmoid')

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs, axis=3)
        out = tf.stack([avg_out, max_out], axis=-1)
        out = self.conv1(out)
        out = out * inputs
        return out
class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg= layers.GlobalAveragePooling2D()
        self.max= layers.GlobalMaxPooling2D()

        self.fc1 = layers.Dense(in_planes//ratio, kernel_initializer='he_normal', activation='relu',
                                kernel_regularizer=regularizers.l2(5e-4),
                                use_bias=True, bias_initializer='zeros')
        self.fc2 = layers.Dense(in_planes, kernel_initializer='he_normal',
                                kernel_regularizer=regularizers.l2(5e-4),
                                use_bias=True, bias_initializer='zeros')

    def call(self, inputs):
        avg_out = self.fc2(self.fc1(self.avg(inputs)))
        max_out = self.fc2(self.fc1(self.max(inputs)))
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)
        out = layers.Reshape((1, 1, out.shape[1]))(out)
        out = out * inputs

        return out

def CBAM(input):
    shape = input.shape
    channel = shape[3]
    CAM = ChannelAttention(channel).call(input)
    out = SpatialAttention().call(CAM)
    return out

#Dimention Transform Block
def D_CBAM_Add(filter_num, input3d, input2d):
    x = Conv3D(1, 1, padding='same', kernel_initializer='he_normal')(input3d)
    x = Lambda(squeeze)(x)
    x = Conv2D(filter_num, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = CBAM(x)
    input2d = CBAM(input2d)
    x = Add()([x, input2d])
    return x



#improved attention gate
def New_AttentionGate_block(s,x, g, inter_channel, i, data_format='channels_last'):
    x_shape = x.shape
    theta_s = Conv2D(inter_channel, [1, 1], strides=[1, 1],data_format=data_format, kernel_regularizer=regularizers.l2(1e-5))(s)
    theta_s = MaxPooling2D(pool_size=(2, 2))(theta_s)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1],dilation_rate=i, data_format=data_format, kernel_regularizer=regularizers.l2(1e-5))(g)
    phi_g = Conv2DTranspose(inter_channel,[2,2],strides=[2,2])(phi_g)

    f = LeakyReLU(alpha=0.2).call(Add()([theta_s, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1],dilation_rate=i, data_format=data_format, kernel_regularizer=regularizers.l2(1e-5))(f)

    sigm_psi_f = Activation(activation='sigmoid')(psi_f)

    att_x = multiply([x, sigm_psi_f])

    return att_x

def New_AttentionGate_block_layer1(s,x, g, inter_channel, i, data_format='channels_last'):
    x_shape = x.shape
    theta_s = Conv2D(inter_channel, [1, 1], strides=[1, 1],data_format=data_format, kernel_regularizer=regularizers.l2(1e-5))(s)

    # theta_s = MaxPooling2D(pool_size=(2, 2))(theta_s)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1],dilation_rate=i, data_format=data_format, kernel_regularizer=regularizers.l2(1e-5))(g)
    print("phi_g shape",phi_g.shape)
    phi_g = Conv2DTranspose(inter_channel,[2,2],strides=[2,2])(phi_g)

    print("s shape:",theta_s.shape)
    print("g.shape",phi_g.shape)
    # f = LeakyReLU(alpha=0.2)(Add([theta_x, phi_g]))
    f = LeakyReLU(alpha=0.2).call(Add()([theta_s, phi_g]))

    psi_f = Conv2D(1, [1, 1], strides=[1, 1],dilation_rate=i, data_format=data_format, kernel_regularizer=regularizers.l2(1e-5))(f)

    sigm_psi_f = Activation(activation='sigmoid')(psi_f)

    # rate = UpSampling2D(size=[2, 2])(sigm_psi_f)
    rate = sigm_psi_f

    att_x = multiply([x, rate])

    return att_x


#our model
def Base_2D_3D_CBAM_ImproRAGD():

    inputs = Input(shape=(192, 192, 4))
    input3d = Lambda(expand)(inputs)
    conv3d1 = Res_BN_block3d(32, input3d)
    pool3d1 = MaxPooling3D(pool_size=2)(conv3d1)

    conv3d2 = Res_BN_block3d(64, pool3d1)
    #
    pool3d2 = MaxPooling3D(pool_size=2)(conv3d2)
    #
    conv3d3 = Res_BN_block3d(128, pool3d2)


    conv1 = Res_BN_block(32, inputs)
    conv1 = D_CBAM_Add(32, conv3d1, conv1)  #pos 1
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Res_BN_block(64, pool1)
    #
    conv2 = D_CBAM_Add(64, conv3d2, conv2)  #pos 2
    #
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Res_BN_block(128, pool2)
    #
    conv3 = D_CBAM_Add(128, conv3d3, conv3)   #pos 3
    #
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Res_BN_block(256, pool3)
    #
    conv4 = Dropout(0.3)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Res_BN_block(512, pool4)
    conv5 = Dropout(0.3)(conv5)
    ########################################
    up6 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    conv4 = New_AttentionGate_block(conv3, conv4, conv5, 256, 1)

    merge6 = Concatenate()([conv4, up6])
    conv6 = Res_BN_block(256, merge6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    conv3 = New_AttentionGate_block(conv2, conv3, conv6, 128, 1)
    merge7 = Concatenate()([conv3, up7])
    conv7 = Res_BN_block(128, merge7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    conv2 = New_AttentionGate_block(conv1, conv2, conv7, 64, 1)
    merge8 = Concatenate()([conv2, up8])
    conv8 = Res_BN_block(64, merge8)
    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    conv1 = New_AttentionGate_block_layer1(inputs, conv1, conv8, 32, 1)
    merge9 = Concatenate()([conv1, up9])
    conv9 = Res_BN_block(32, merge9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    return model