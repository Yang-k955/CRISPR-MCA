
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.layers import Add, GRU,Embedding,Activation,ReLU,AveragePooling2D,MaxPool2D,MaxPooling1D,BatchNormalization,Conv1D,\
    Attention, Dense, Conv2D, Bidirectional, LSTM, Flatten, Input, Activation, Reshape, Dropout, Concatenate, AveragePooling1D, MaxPool1D, \
    BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, GRU, AdditiveAttention, AlphaDropout, LeakyReLU
from tensorflow.keras.initializers import VarianceScaling,RandomUniform
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score
from keras_layer_normalization import LayerNormalization
from tensorflow.keras.initializers import glorot_normal, HeNormal
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import multiply, concatenate
from tensorflow.python.keras.layers.core import Dense, Dropout, Lambda, Flatten, Reshape, Permute
from tensorflow.keras import layers, models

def CnnCrispr():
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    input = Input(shape=(23,))
    embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,trainable=True)(input)
    bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
    bi_lstm_relu = Activation('relu')(bi_lstm)

    conv1 = Conv1D(10, (5))(bi_lstm_relu)
    conv1_relu = Activation('relu')(conv1)
    conv1_batch = BatchNormalization()(conv1_relu)

    conv2 = Conv1D(20, (5))(conv1_batch)
    conv2_relu = Activation('relu')(conv2)
    conv2_batch = BatchNormalization()(conv2_relu)

    conv3 = Conv1D(40, (5))(conv2_batch)
    conv3_relu = Activation('relu')(conv3)
    conv3_batch = BatchNormalization()(conv3_relu)

    conv4 = Conv1D(80, (5))(conv3_batch)
    conv4_relu = Activation('relu')(conv4)
    conv4_batch = BatchNormalization()(conv4_relu)

    conv5 = Conv1D(100, (5))(conv4_batch)
    conv5_relu = Activation('relu')(conv5)
    conv5_batch = BatchNormalization()(conv5_relu)

    flat = Flatten()(conv5_batch)
    drop = Dropout(0.3)(flat)
    dense = Dense(20)(drop)
    dense_relu = Activation('relu')(dense)
    prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
    model = Model(inputs=[input], outputs=[prediction])
    return model

def CRISPR_Net():
    inputs = Input(shape=(1, 24, 7), name='main_input')
    branch_0 = Conv2D(10, (1, 1), strides=1, padding='same', use_bias=True,
                      kernel_initializer=glorot_normal(seed=2023), name=None, trainable=True)(inputs)
    branch_1 = Conv2D(10, (1, 2), strides=1, padding='same', use_bias=True,
                      kernel_initializer=glorot_normal(seed=2023), name=None, trainable=True)(inputs)
    branch_2 = Conv2D(10, (1, 3), strides=1, padding='same', use_bias=True,
                      kernel_initializer=glorot_normal(seed=2023), name=None, trainable=True)(inputs)
    branch_3 = Conv2D(10, (1, 4), strides=1, padding='same', use_bias=True,
                      kernel_initializer=glorot_normal(seed=2023), name=None, trainable=True)(inputs)

    branches = [inputs, branch_0, branch_1, branch_2, branch_3]
    mixed = concatenate(branches, axis=-1)
    mixed = Reshape((24, 47))(mixed)
    blstm_out = Bidirectional(
        LSTM(15, kernel_initializer=glorot_normal(seed=2023), return_sequences=True, input_shape=(24, 47),
             name="LSTM_out"))(mixed)
    # inputs_rs = Reshape((24, 7))(inputs)
    # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
    blstm_out = Flatten()(blstm_out)
    x = Dense(80, kernel_initializer=glorot_normal(seed=2023), activation='relu')(blstm_out)
    x = Dense(20, kernel_initializer=glorot_normal(seed=2023), activation='relu')(x)
    x = Dropout(0.35, seed=2023)(x)
    prediction = Dense(2, kernel_initializer=glorot_normal(seed=2023), activation='softmax', name='main_output')(x)
    model = Model(inputs=[inputs], outputs=[prediction])
    return model



def residual_block(x, num_filters):
    res = x
    x = Conv1D(num_filters, kernel_size=5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv1D(num_filters, kernel_size=5, padding='same', activation='relu')(x)
    x = Conv1D(num_filters, kernel_size=5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([res, x])
    x = Activation('relu')(x)
    return x

def transformer_encoder_block(input_tensor, head_num, dff, rate=0.1):
    """Builds a transformer encoder block."""
    attention_output = MultiHeadAttention(head_num=head_num)([input_tensor, input_tensor, input_tensor])
    attention_output = Dropout(rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + input_tensor)

    ffn_output = Dense(dff, activation='relu')(attention_output)
    ffn_output = Dense(input_tensor.shape[-1])(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)

    return ffn_output

def attention(x, g, TIME_STEPS):
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """
    input_dim = int(x.shape[2])
    x1 = K.permute_dimensions(x, (0, 2, 1))
    g1 = K.permute_dimensions(g, (0, 2, 1))

    x2 = Reshape((input_dim, TIME_STEPS))(x1)
    g2 = Reshape((input_dim, TIME_STEPS))(g1)

    x3 = Dense(TIME_STEPS, kernel_initializer=RandomUniform(seed=2020))(x2)
    g3 = Dense(TIME_STEPS, kernel_initializer=RandomUniform(seed=2020))(g2)
    x4 = tf.keras.layers.add([x3, g3])
    a = Dense(TIME_STEPS, activation="softmax", use_bias=False)(x4)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = multiply([x, a_probs])
    return output_attention_mul

def CRISPR_OFFT():
    VOCAB_SIZE = 16
    EMBED_SIZE = 90
    MAXLEN = 23
    input = Input(shape=(23,))
    embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

    conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
    batchnor1 = BatchNormalization()(conv1)

    conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
    batchnor2 = BatchNormalization()(conv2)

    conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
    batchnor3 = BatchNormalization()(conv3)

    conv11 = Conv1D(80, 9, name="conv11")(batchnor1)

    x = Attention()([conv11, batchnor3])

    flat = Flatten()(x)
    dense1 = Dense(40, activation="relu", name="dense1")(flat)
    drop1 = Dropout(0.2)(dense1)

    dense2 = Dense(20, activation="relu", name="dense2")(drop1)
    drop2 = Dropout(0.2)(dense2)

    output = Dense(2, activation="softmax", name="dense3")(drop2)
    model = Model(inputs=[input], outputs=[output])
    return model


def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=True,
              name=None, trainable=True):
    x = layers.Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name, trainable=trainable)(x)

    # x = layers.BatchNormalization(axis=-1,scale=True)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = layers.Activation(activation, name=ac_name)(x)
    return x

def CRISPR_IP():
    inputs = Input(shape=(1, 24, 7), name='main_input')
    conv_1_output = Conv2D(60, (1, 7), padding='valid')(inputs)
    conv_1_output_reshape = Reshape((18, 60))(conv_1_output)
    #conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
    conv_1_output_reshape2 = Permute((2, 1))(conv_1_output_reshape)
    conv_1_output_reshape_average = AveragePooling1D(pool_size=2, strides=None, padding='valid')(conv_1_output_reshape2)
    conv_1_output_reshape_max = MaxPool1D(pool_size=2, strides=None, padding='valid')(conv_1_output_reshape2)
    bidirectional_1_output = Bidirectional(LSTM(30, return_sequences=True, dropout=0.25))\
        (concatenate([conv_1_output_reshape_average, conv_1_output_reshape_max],axis=-1))
    attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
    average_1_output = AveragePooling1D(pool_size=2, strides=None, padding='valid')(attention_1_output)
    max_1_output = MaxPool1D(pool_size=2, strides=None, padding='valid')(attention_1_output)

    concat_output = concatenate([average_1_output, max_1_output],axis=-1)

    flatten_output = Flatten()(concat_output)
    linear_1_output = BatchNormalization()(
        Dense(200, activation='relu')(flatten_output))
    linear_2_output = Dense(100, activation='relu')(linear_1_output)
    linear_2_output_dropout = Dropout(0.9)(linear_2_output)
    output = Dense(2, activation='softmax')(linear_2_output_dropout)
    model = Model(inputs=[inputs], outputs=[output])
    return model

def cnn_std():
    input = Input(shape=(1, 23, 4))
    conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
    conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
    conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
    conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

    conv_output = concatenate([conv_1, conv_2, conv_3, conv_4])

    bn_output = BatchNormalization()(conv_output)

    pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

    flatten_output = Flatten()(pooling_output)

    x = Dense(100, activation='relu')(flatten_output)
    x = Dense(23, activation='relu')(x)
    x = Dropout(rate=0.15)(x)

    output = Dense(2, activation="softmax")(x)

    model = Model(inputs=[input], outputs=[output])
    return model

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None,**kwargs):
        super(PositionalEncoding, self).__init__()
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim

    def call(self, x):
        position_embedding = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
            for pos in range(self.sequence_len)])

        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # dim 2i
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # dim 2i+1
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)
        return position_embedding+x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sequence_len': self.sequence_len,
            'embedding_dim': self.embedding_dim
        })
        return config


def CRISPR_DNT():

    initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
    input_value = Input(shape=(23, 14, 1))

    # 卷积
    conv_1_output = Conv2D(64, (1, 14), activation='relu', kernel_initializer=initializer, data_format='channels_first')(input_value)
    conv_1_output = BatchNormalization()(conv_1_output)

    conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
        conv_1_output)
    #conv_1_output_reshape2 = Permute((2, 1))(conv_1_output_reshape)
    conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])

    # 池化
    conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
    conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
    input_value1 = Reshape((23,14))(input_value)

    #  BLSTM
    bidirectional_1_output = Bidirectional(
        LSTM(32, return_sequences=True, dropout=0.2,activation='relu',kernel_initializer=initializer))(
        Concatenate(axis=-1)([input_value1,conv_1_output_reshape_average, conv_1_output_reshape_max]))
    bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)

    # Multi-head Self-attention
    pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
    attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
    residual1 = attention_1_output + pos_embedding
    laynorm1 = LayerNormalization()(residual1)
    linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
    linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
    residual2 = laynorm1 + linear2

    #  2次 Multi-head Self-attention
    laynorm2 = LayerNormalization()(residual2)
    attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
    residual3 = attention_2_output + laynorm2
    laynorm3 = LayerNormalization()(residual3)
    linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
    linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
    residual4 = laynorm3 + linear4

    #  Dense Layer
    laynorm4 = LayerNormalization()(residual4)
    flatten_output = Flatten()(laynorm4)
    linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
    # linear_1_output = BatchNormalization()(linear_1_output)
    # linear_1_output = Dropout(0.25)(linear_1_output)
    linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
    linear_2_output_dropout = Dropout(0.25)(linear_2_output)
    linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
    model = Model(input_value, linear_3_output)
    return model


if __name__ == '__main__':
    mdoel = CRISPR_DNT()
    mdoel.summary()
