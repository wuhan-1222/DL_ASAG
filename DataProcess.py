import numpy as np
import pandas as pd
import xlrd
import os

from Paraments import args

question_set = xlrd.open_workbook(args.QuestionData)
score_set = xlrd.open_workbook(args.ScoreData)

# score_sh = score_set.sheet_by_name('主观题评分语料7896')
define_sh = score_set.sheet_by_name('定义类评分语料2754')
common_sh = score_set.sheet_by_name('一般类评分语料2568')
list_sh = score_set.sheet_by_name('顺序类评分语料2574')

question_sh = question_set.sheet_by_name('问句分类语料3948')

# print(score_sh.row_values(1)) # ['定义类', '简述接入控制的定义。', '接入控制的定义如下：一个强加允许用户访问网络资源的控制方法，通常基于用户的账户。', '定义如下：一个强加允许用户访问网络资源的控制方法，通常基于用户的账户。', 10.0]

def Divi(sentence):
    sent = [sentence[w] for w in range(len(sentence))]
    sent = ' '.join(sent)
    return sent

# with open('./datasets/Score.txt', encoding='utf-8', mode='a') as s:
#
#     for i in range(score_sh.nrows):
#
#         question = Divi(score_sh.row_values(i)[1])
#         reference = Divi(score_sh.row_values(i)[2])
#         answer = Divi(score_sh.row_values(i)[3])
#         score = str(score_sh.row_values(i)[4])
#
#         s.writelines([question, '\t' ,reference, '\t', answer, '\t', score, '\n'])
#     s.close()


with open('./datasets/Define.txt', encoding='utf-8', mode='a') as d:
    for i in range(define_sh.nrows):

        question = Divi(define_sh.row_values(i)[1])
        reference = Divi(define_sh.row_values(i)[2])
        answer = Divi(define_sh.row_values(i)[3])
        score = str(define_sh.row_values(i)[4])

        d.writelines([question, '\t' ,reference, '\t', answer, '\t', score, '\n'])
    d.close()

with open('./datasets/Common.txt', encoding='utf-8', mode='a') as c:
    for i in range(common_sh.nrows):

        question = Divi(common_sh.row_values(i)[1])
        reference = Divi(common_sh.row_values(i)[2])
        answer = Divi(common_sh.row_values(i)[3])
        score = str(common_sh.row_values(i)[4])

        c.writelines([question, '\t' ,reference, '\t', answer, '\t', score, '\n'])
    c.close()

with open('./datasets/List.txt', encoding='utf-8', mode='a') as l:
    for i in range(list_sh.nrows):

        question = Divi(list_sh.row_values(i)[1])
        reference = Divi(list_sh.row_values(i)[2])
        answer = Divi(list_sh.row_values(i)[3])
        score = str(list_sh.row_values(i)[4])

        l.writelines([question, '\t' ,reference, '\t', answer, '\t', score, '\n'])
    l.close()


# with open('./datasets/Question.txt', encoding='utf-8', mode='a') as q:
#
#     for i in range(question_sh.nrows):
#         question = Divi(question_sh.row_values(i)[1])
#         q.writelines([question_sh.row_values(i)[0], '\t' ,question, '\n'])
#     q.close()