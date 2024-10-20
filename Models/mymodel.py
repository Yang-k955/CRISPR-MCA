import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.layers import Add, GRU,Embedding,Activation,ReLU,AveragePooling2D,MaxPool2D,MaxPooling1D,BatchNormalization,Conv1D,\
    Attention, Dense, Conv2D, Bidirectional, LSTM, MaxPooling2D, Flatten, Input, Activation, Reshape, Dropout, Concatenate, AveragePooling1D, MaxPool1D, \
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

class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, max_seq_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.pos_encoding = self.get_positional_encoding(self.max_seq_length, self.d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        return x + self.pos_encoding[:length, :]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'max_seq_length': self.max_seq_length,
            'd_model': self.d_model,
        })
        return config

    def get_positional_encoding(self, max_seq_length, d_model):
        positional_encoding = np.zeros((max_seq_length, d_model))
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                positional_encoding[pos, i] = np.sin(pos / np.power(10000, (2 * (i // 2)) / d_model))
                positional_encoding[pos, i + 1] = np.cos(pos / np.power(10000, (2 * (i // 2)) / d_model))
        return tf.cast(positional_encoding, dtype=tf.float32)

def MCNN(input_tensor, filter_channels):

    branch1 = Conv2D(filter_channels, (1, 1), padding='same', activation='relu')(input_tensor)

    branch2 = Conv2D(filter_channels, (1, 1), padding='same', activation='relu')(input_tensor)
    branch2 = Conv2D(filter_channels, (3, 3), padding='same', activation='relu')(branch2)

    branch3 = Conv2D(filter_channels, (1, 1), padding='same', activation='relu')(input_tensor)
    branch3 = Conv2D(filter_channels, (5, 5), padding='same', activation='relu')(branch3)

    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
    branch4 = Conv2D(filter_channels, (1, 1), padding='same', activation='relu')(branch4)

    output = concatenate([branch1, branch2, branch3, branch4], axis=-1)
    return output

def CRISPR-MCA():
    inputs = Input(shape=(1, 24, 7), name='main_input')
    inception_output = MCNN(inputs, 10)
    mixed = Reshape((24, -1))(inception_output)
    mixed = BatchNormalization()(mixed)

    pos_encoding_layer = PositionalEncodingLayer(24, 40)  
    x = pos_encoding_layer(mixed)
    attention_output = MultiHeadAttention(head_num=8)([x, x, x]) 
    attention_output = Dropout(0.2)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + x)

    blstm_out = Flatten()(attention_output)

    x = Dense(256, activation='relu')(blstm_out)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    prediction = Dense(2, activation='softmax', name='main_output')(x)

    model = Model(inputs=[inputs], outputs=[prediction])
    return model

if __name__ == '__main__':
    mdoel = CRISPR-MCA()
    mdoel.summary()
    pass
