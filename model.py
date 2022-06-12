from tensorflow.keras.layers import Input, LSTM, Bidirectional,Concatenate
from layers import *
from tensorflow.keras.models import Model
from Bi_Attention import BiAttentionLayer
from Paraments import args

def Score(embedding_matrix, word_index, max_sequence_length,  word_embedding_dim):
    # =========== 参数 ================
    nb_words = min(200000, len(word_index)) + 1
    context_rnn_dim = 100
    dense_dim = 100


    print('===========build model============')
    w1 = Input(shape=(max_sequence_length,), dtype='int32')
    w2 = Input(shape=(max_sequence_length,), dtype='int32')

    word_layer = WordRepresLayer(
        max_sequence_length, nb_words, word_embedding_dim, embedding_matrix)
    w_res1 = word_layer(w1)  # [batch_size,sequence_length,embedding_dim]
    w_res2 = word_layer(w2)

    # cnn_layer = CNNLayer(input_shape=(K.int_shape(w_res1)[1]))
    # conv1 = cnn_layer(w_res1)
    # conv2 = cnn_layer(w_res2)

    context_layer = ContextLayer(
        context_rnn_dim, dropout=0.1,
        input_shape=(K.int_shape(w_res1)[1], K.int_shape(w_res1)[-1],), return_sequence=True)
    context1 = context_layer(w_res1)
    context2 = context_layer(w_res2)

    cnn = layers.concatenate([context1, context2], axis=1)
    decorate = tf.reduce_mean(cnn, axis=-1)
    pred = PredictLayer(dense_dim,
                        input_dim=K.int_shape(decorate)[-1],
                        dropout=0.1)(decorate)

    # pred = ESIM(context1, context2)
    # pred = ESIM(w_res1, w_res2)

    model = Model(inputs=(w1, w2), outputs=pred)
    return model

def ESIM(a, b):
    co_attention_matrix = BiAttentionLayer(K.int_shape(a)[-1], K.int_shape(a)[1], K.int_shape(b)[1])([a, b])
    ea = layers.Softmax(axis=2)(co_attention_matrix)
    eb = layers.Softmax(axis=1)(co_attention_matrix)
    e1 = tf.expand_dims(ea, axis=-1)  # [batch,seq,seq,1]
    e2 = tf.expand_dims(eb, axis=-1)  # [batch,seq,seq,1]

    x1 = tf.expand_dims(b, axis=1)  # [batch,1,seq_b,emb]
    x1 = layers.Multiply()([e1, x1])  # [batch,seq_a,seq_b,emb]
    x1 = tf.reduce_sum(x1, axis=2)  # [batch,seq_a,emb]
    x2 = tf.expand_dims(a, axis=2)  # [batch,seq_a,1,emb]
    x2 = layers.Multiply()([e2, x2])
    x2 = tf.reduce_sum(x2, axis=2)

    m1 = layers.Concatenate()([a, x1, layers.Subtract()([a, x1]), layers.Multiply()([a, x1])])
    m2 = layers.Concatenate()([b, x2, layers.Subtract()([b, x2]), layers.Multiply()([b, x2])])

    y1 = layers.Bidirectional(layers.LSTM(300, return_sequences=True))(m1)
    y2 = layers.Bidirectional(layers.LSTM(300, return_sequences=True))(m2)

    mx1 = layers.Lambda(K.max, arguments={'axis': 1})(y1)
    av1 = layers.Lambda(K.mean, arguments={'axis': 1})(y1)
    mx2 = layers.Lambda(K.max, arguments={'axis': 1})(y2)
    av2 = layers.Lambda(K.mean, arguments={'axis': 1})(y2)

    y = layers.Concatenate()([av1, mx1, av2, mx2])
    y = layers.Dense(1024, activation='relu')(y)
    y = layers.Dropout(0.1)(y)
    # y = layers.Dense(1024, activation='tanh')(y)
    # y = layers.Dropout(0.1)(y)
    y = layers.Dense(len(args.Slabel), activation='softmax')(y)
    return y