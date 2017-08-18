#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Input,Activation,Embedding,Flatten,merge,Merge
from keras.utils import np_utils
from keras.layers.convolutional import  MaxPooling2D,MaxPooling1D,Convolution1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.layers.wrappers import Bidirectional
from keras.preprocessing import  sequence
from keras.layers.core import Dropout,Lambda
from keras.regularizers import l1,l2,activity_l1,activity_l2

# 导入sklear的数据shuffle
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle

timestep = 80    #每条语料设定的最大长度
embedding_length = 100   #embedding的维度
lexical_words = 12     #词汇级别的单词数   4*3, 最多2个人名,每个人名包括前后共6个词
lexical_length = embedding_length * lexical_words   # 词级别的特征向量长度
pos_embedding_length = 10   #位置表的的embedding长度
#句子级别每个单词窗口大小
sentence_windows = 3
#句子级别的embedding总长度
sentence_length = embedding_length * sentence_windows + 2 * pos_embedding_length
label_count = 7
classWeight = {}   #类别权重，处理类别数据不平衡用

def class_weight():
    # 'man_and_wife': 0,
    # 'parent_children': 1,
    # 'teacher_and_pupil': 2,
    # 'cooperate': 3,
    # 'friends': 4,
    # 'brother': 5,
    # 'sweetheart': 6
    global classWeight
    relationsNumber = [1395,1451,1030,537,207,225,328]
    weight = [1.0*number/100 for number in relationsNumber]
    multiplySum = 1.0
    for w in weight:
        multiplySum *= w
    for i in range(len(weight)):
        weight[i] = multiplySum / weight[i]
    minWeight = min(weight)
    dec = 1
    while minWeight > 1:
        dec *= 10
        minWeight /= 10
    for i in range(len(weight)):
        # 小数点保留三位
        weight[i] = round(weight[i] / dec,3)
        classWeight[i] = weight[i]

def read_train_data():

    train_lexical_data = np.load("../paper_cnn_train_lexical_data.npy")
    train_sentence_data = np.load("../paper_cnn_train_sentence_data.npy")
    train_label = np.load("../paper_cnn_train_label.npy")
    return train_lexical_data,train_sentence_data,train_label

def read_test_data():

    test_lexical_data = np.load("../paper_cnn_test_lexical_data.npy")
    test_sentence_data = np.load("../paper_cnn_test_sentence_data.npy")
    test_label = np.load("../paper_cnn_test_label.npy")
    return test_lexical_data,test_sentence_data,test_label

def cnn(train_lexical_data,train_sentence_data,train_label,
         test_lexical_data,test_sentence_data,test_label):

    train_label = np_utils.to_categorical(train_label,label_count)
    test_label_categorical =  np_utils.to_categorical(test_label,label_count)

    lexical_branch = Sequential()
    lexical_branch.add(Dense(200,activation='tanh',input_dim=lexical_length,
                            W_regularizer=l2(0.01)))
    sentence_branch = Sequential()
    sentence_branch.add(Convolution1D(200,3, border_mode='same', W_regularizer=l2(0.01),
                                      input_shape=(timestep, sentence_length)))
    sentence_branch.add(GlobalMaxPooling1D())
    # sentence_branch.add(Dropout(0.5))
    sentence_branch.add(Dense(100,activation='tanh'))

    merged = Merge([lexical_branch,sentence_branch],mode='concat')
    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(label_count,activation='softmax',name='output'))

    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    final_model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

    epoch = 50
    for i in range(epoch):
        print i,'/',epoch
        final_model.fit([train_lexical_data, train_sentence_data],train_label, batch_size=64,
                        nb_epoch=1, shuffle=True,verbose=1,validation_data=
                        ([test_lexical_data,test_sentence_data],test_label_categorical))

        test_classes = final_model.predict_classes([test_lexical_data,test_sentence_data],batch_size=64)
        putOut(test_classes,test_label)

def putOut(test_classes,test_label):
    list_actual = [0] * label_count  #实际属于每个关系类别的个数
    list_correct = [0] * label_count   #被正确抽取的属于每个关系类别的个数
    list_all_extract = [0] * label_count  #所有抽取为每个关系类别的数

    for item in test_label:
        list_actual[item[0]] += 1

    for i in range(len(test_classes)):
        predict_class = test_classes[i]

        list_all_extract[predict_class] += 1
        if predict_class ==test_label[i][0]:
            list_correct[predict_class] += 1

    precision = [0] * label_count
    recall = [0] * label_count
    fscore = [0] *label_count
    sum_fscore = 0
    for i in range(label_count):
        try:
            precision[i] = 1.0 * list_correct[i] / list_all_extract[i]
        except: precision[i] = 0
        try:
            recall[i] = 1.0 * list_correct[i] /list_actual[i]
        except: recall[i] = 0
        try:
            fscore[i] = 2.0 * precision[i] * recall[i] / (precision[i] + recall[i])
        except: fscore[i] = 0

        sum_fscore += fscore[i]
    print 'average fscore: ',sum_fscore / label_count

if __name__ == "__main__":

     train_lexical_data,train_sentence_data,train_label = read_train_data()
     test_lexical_data,test_sentence_data,test_label = read_test_data()

     class_weight()

     cnn(train_lexical_data,train_sentence_data,train_label,
         test_lexical_data,test_sentence_data,test_label)


