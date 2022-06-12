import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
import numpy as np
import six
import math
from Paraments import args


class WordRepresLayer(object):
    '''
    Word embedding representation layer
    '''
    def __init__(self,sequence_length,nb_words,word_embedding_dim,embedding_matrix): # inputs = [batch_size,sequence_length]
        self.model = keras.Sequential()
        self.model.add(layers.Embedding(nb_words,
                                        word_embedding_dim,
                                        weights=[embedding_matrix],
                                        input_length=sequence_length,
                                        trainable=False,
                                        mask_zero=True)) # [batch_size,sequence_length,embedding_dim]

    def __call__(self, inputs):
        return self.model(inputs)

class CharRepresLayer(object):
    '''
    Char embedding representation layer
    '''
    def __init__(self,sequence_length,nb_chars,nb_per_word,embedding_dim,
            rnn_dim,dropout=0.0):
        def _collapse_input(x,nb_per_word=0): # x = [batch_size,sequence_length,nb_per_word]
            x = K.reshape(x,(-1,nb_per_word))
            return x # x = [batch_size * sequence_length,nb_per_word]

        def _unroll_input(x,sequence_length=0,rnn_dim=0):
            x = K.reshape(x,(-1,sequence_length,rnn_dim))
            return x

        self.model = keras.Sequential()
        self.model.add(layers.Lambda(_collapse_input,
                                     arguments={'nb_per_word':nb_per_word},
                                     output_shape=(nb_per_word,),
                                     input_shape=(sequence_length,nb_per_word,))) # [batch_size * sequence_length,nb_per_word]
        self.model.add(layers.Embedding(nb_chars,
                                        embedding_dim,
                                        input_length=nb_per_word,
                                        trainable=True)) # [batch_size * sequence_length,nb_per_word,embedding_dim]
        self.model.add(layers.LSTM(rnn_dim,
                           dropout=dropout,
                           recurrent_dropout=dropout)) # [batch_size * sequence_length,rnn_dim]
        self.model.add(layers.Lambda(_unroll_input,
                                     arguments={'sequence_length':sequence_length,
                                                'rnn_dim':rnn_dim},
                                     output_shape=(sequence_length,rnn_dim))) # [batch_size,sequence_length,rnn_dim]
    def __call__(self, inputs):
        return self.model(inputs)

class CharRepresLayerConv(object):
    def __init__(self,sequence_length,nb_chars,nb_per_word,embedding_dim,filters):
        def _collapse_input(x,nb_per_word=0): # x = [batch_size,sequence_length,nb_per_word]
            x = K.reshape(x,(-1,nb_per_word))
            return x # x = [batch_size * sequence_length,nb_per_word]
        def _reshape_dims(x,nb_per_word,embedding_dim):
            x = K.reshape(x,(-1,nb_per_word,embedding_dim,1))
            return x

        def _extract_dims(x,sequence_len,nb_per_word,filters):
            x = K.reshape(x,(-1,sequence_len*(nb_per_word),filters))
            return x

        self.model = keras.Sequential()
        self.model.add(layers.Lambda(_collapse_input,
                                     arguments={'nb_per_word': nb_per_word},
                                     output_shape=(nb_per_word,),
                                     input_shape=(
                                     sequence_length, nb_per_word,)))  # [batch_size * sequence_length,nb_per_word]
        self.model.add(layers.Embedding(nb_chars,
                                        embedding_dim,
                                        input_length=nb_per_word,
                                        trainable=True))  # [batch_size * sequence_length,nb_per_word,embedding_dim]
        self.model.add(layers.Lambda(_reshape_dims,
                                     arguments={'nb_per_word':nb_per_word,
                                                'embedding_dim':embedding_dim},
                                     output_shape=(nb_per_word,embedding_dim,1)))
        self.model.add(layers.Conv2D(filters=filters,kernel_size=[3,embedding_dim],strides=[1,1],padding='VALID',
                                     data_format='channels_last',use_bias=True,activation='relu'))
        self.model.add(layers.Permute([1,3,2]))
        self.model.add(layers.Lambda(_extract_dims,
                                     arguments={'sequence_len':sequence_length,
                                                'nb_per_word':nb_per_word-2,
                                                'filters':filters},
                                     output_shape=(sequence_length*(nb_per_word-2),filters)))
        self.model.add(layers.MaxPooling1D(pool_size=10,strides=10,padding='VALID'))
    def __call__(self, inputs):
        return self.model(inputs)

class CNNLayer(object): # [batch_size,sequence_length,embedding_dim]或者[batch_size,sequence_length,embedding_dim + rnn_dim]
    '''
    CNN layer
    '''
    def __init__(self,input_shape=(0,)):
        self.model = keras.Sequential()
        # self.model.add(layers.Reshape([input_shape[0],input_shape[1],1]))
        self.model.add(layers.Conv1D(filters=300,kernel_size=3,padding='SAME',
                                     data_format='channels_last',use_bias=True,activation='relu')) #[3,embedding_dim]

    def __call__(self, inputs):
        return self.model(inputs) # output_shape=[]


class ContextLayer(object): # [batch_size,sequence_length,embedding_dim + rnn_dim]
    '''
    Word context representation layer
    '''
    def __init__(self,rnn_dim,input_shape=(0,),
                 dropout=0.0,return_sequence=False):

        self.model = keras.Sequential()
        self.model.add(layers.Bidirectional(layers.LSTM(rnn_dim,
                                                activation='tanh',
                                                return_sequences=return_sequence,
                                                kernel_regularizer=keras.regularizers.l2(0.01)),# return_sequences=True => [batch_size,rnn_dim]
                                            input_shape=input_shape)) # Bidirectional => [batch_size,2*rnn_dim]
        # self.model.add(layers.MaxPooling1D())
        self.model.add(layers.Dropout(dropout))
        # self.model.add(LayerNormalization(hidden_units=4*rnn_dim))
        self.model.add(layers.BatchNormalization())

    def __call__(self, inputs):
        return self.model(inputs)

class PredictLayer(object):
    '''
    Prediction layer
    '''
    def __init__(self,dense_dim,input_dim=0,dropout=0.0):
        self.model = keras.Sequential()
        self.model.add(layers.Dense(dense_dim,
                                    activation='relu',
                                    # activation = 'tanh',
                                    input_shape=(input_dim,)))
        self.model.add(layers.Dropout(dropout))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(len(args.Slabel),activation='sigmoid'))
    def __call__(self, inputs):
        return self.model(inputs)