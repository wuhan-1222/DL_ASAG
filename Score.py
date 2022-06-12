from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import  precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error

from Fuction import *
from model import *
from Paraments import args

def GetData(filepath, labels):
    print('Loading data from ' + filepath)
    q = []
    r = []
    a = []
    s = []

    f = open(filepath, encoding='utf-8')
    next(f)
    for line in f:
        d = line.strip().split('\t')
        if len(d) < 4:
            continue
        if d[3] in args.Slabel.keys():
            q.append(d[0])
            r.append(d[1])
            a.append(d[2])
            s.append(labels[d[3]])
    f.close()

    q = np.asarray(q)
    r = np.asarray(r)
    a = np.asarray(a)
    s = np.asarray(s)

    shuffle_indices = np.random.permutation(np.arange(len(q)))
    train_q = q[shuffle_indices][:int(0.8 * len(q))]
    train_r = r[shuffle_indices][:int(0.8 * len(q))]
    train_a = a[shuffle_indices][:int(0.8 * len(q))]
    train_s = s[shuffle_indices][:int(0.8 * len(q))]

    dev_q = q[shuffle_indices][int(0.8 * len(q)):]
    dev_r = r[shuffle_indices][int(0.8 * len(q)):]
    dev_a = a[shuffle_indices][int(0.8 * len(q)):]
    dev_s = s[shuffle_indices][int(0.8 * len(q)):]
    return train_q, train_r, train_a, train_s, dev_q, dev_r, dev_a, dev_s, q, r, a

train_q, train_r, train_a, train_s, dev_q, dev_r, dev_a, dev_s, q, r, a = GetData(args.train, args.Slabel)


tk = Tokenizer(num_words=40000)
tk.fit_on_texts(q)
word_index = tk.word_index

# print(len(work_index))
train_q = tk.texts_to_sequences(train_q)
train_q = pad_sequences(train_q, maxlen=args.MAX_LEN, padding='post')
train_r = tk.texts_to_sequences(train_r)
train_r = pad_sequences(train_r, maxlen=args.MAX_LEN, padding='post')
train_a = tk.texts_to_sequences(train_a)
train_a = pad_sequences(train_a, maxlen=args.MAX_LEN, padding='post')
#
dev_q = tk.texts_to_sequences(dev_q)
dev_q = pad_sequences(dev_q, maxlen=args.MAX_LEN, padding='post')
dev_r = tk.texts_to_sequences(dev_r)
dev_r = pad_sequences(dev_r, maxlen=args.MAX_LEN, padding='post')
dev_a = tk.texts_to_sequences(dev_a)
dev_a = pad_sequences(dev_a, maxlen=args.MAX_LEN, padding='post')

# 读取词向量文件
embedding_matrix = load_glove_matrix(args.Word2Vec,word_index, args.WORD_EMBEDDING_DIM)

def scheduler(epoch):
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr,lr + 1e-4)
        print('lr changed to {}'.format(lr + 1e-4))
    return K.get_value(model.optimizer.lr)

model = Score(embedding_matrix, word_index, args.MAX_LEN*2, args.WORD_EMBEDDING_DIM)
model.compile(optimizer=keras.optimizers.Adam(lr=1e-4,clipvalue=5),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

train_qr = np.hstack((train_q, train_r))
train_qa = np.hstack((train_q, train_a))
dev_qr = np.hstack((dev_q, dev_r))
dev_qa = np.hstack((dev_q, dev_a))
# print(train_qr.shape)

model.fit([train_qr, train_qa],train_s,
          batch_size=args.batch_size,
          validation_split=0.1,
          epochs=args.epoch)
_,acc = model.evaluate([dev_qr, dev_qa], dev_s,batch_size=args.batch_size)

y_pred = model.predict([dev_qr, dev_qa], batch_size=args.batch_size).argmax(axis=1)

# model.summary()

def pearson(vector1, vector2):
    n = len(vector1)
    #simple sums
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    #sum up the products
    p_sum = sum([vector1[i]*vector2[i] for i in range(n)])
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den

rmse = math.sqrt(mean_squared_error(dev_s,y_pred))
mae = mean_absolute_error(dev_s,y_pred)
p = pearson(dev_s, y_pred)
f1 = f1_score(dev_s,y_pred,average='macro')
pre = precision_score(dev_s,y_pred,average='macro')
rec = recall_score(dev_s, y_pred, average='macro')
print("acc: {:.3f}".format(acc), "f1: {:.3f}".format(f1), "pre: {:.3f}".format(pre) , "rec: {:.3f}".format(rec),
      "mae: {:.3f}".format(mae), "rmse: {:.3f}".format(rmse), "pearson: {:.3f}".format(p))
