class args:

    ScoreData = './datasets/主观题评分语料.xlsx'
    QuestionData = './datasets/问句分类语料.xlsx'

    train = './kfold_data/4/train.txt'
    test = './kfold_data/4/test.txt'

    train_score_define = './datasets/Define.txt'
    train_score_common = './datasets/Common.txt'
    train_score_list = './datasets/List.txt'

    labels = {'定义类':0, '顺序类':1, '一般类':2}
    # Slabel = {'0.0':0, '2.0':1, '2.5':1, '3.0':1, '4.0':2, '5.0':2, '6.0':3, '7.5':4, '10.0':5}
    Slabel = {'0':0,'0.5':1,'1':2,'1.5':3,'2':4,'2.5':5,'3':6,
           '3.5':7,'4':8,'4.5':9,'5':10}
    # Word2Vec = './datasets/token_vec_300.bin'
    Word2Vec = 'crawl-300d-2M.vec'
    MAX_LEN = 12
    WORD_EMBEDDING_DIM = 300
    epoch = 20
    batch_size = 32
