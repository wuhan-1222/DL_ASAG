import numpy as np

def load_glove_matrix(vec_file,word_index,WORD_EMBEDDING_DIM):
    print('---- loading glove ...')
    embedding_index = {}
    f = open(vec_file,'rt',encoding='utf-8')
    next(f)
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embedding_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embedding_index))

    nb_words = len(word_index) + 1
    embedding_matrix = np.zeros((nb_words,WORD_EMBEDDING_DIM))
    for word,i  in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def combine(x,y):
    return np.vstack((x, y))

def ALL(a,b,c):
    d = combine(a,b)
    return np.vstack((d, c))