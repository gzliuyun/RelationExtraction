#__author__ = 'Administrator'
# -*- coding: utf-8 -*-

######################
##                  ##
##   最终确定的模型    ##
##                  ##
######################
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Input,Activation,Embedding,Flatten,merge,Merge,SimpleRNN,LSTM
from keras.utils import np_utils
from keras.layers.convolutional import  MaxPooling2D,MaxPooling1D,Convolution1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.layers.wrappers import Bidirectional
from keras.preprocessing import  sequence
from keras.layers.core import Dropout,Lambda
from keras.regularizers import l1,l2

# 导入sklear的数据shuffle
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle

timestep = 80    #每条语料设定的最大长度
embedding_length = 100   #embedding的维度
rnn_step = 20     #rnn部分的单词数   4*5, 最多4个人名,每个人名包括前后共5个词
window = 3  # cnn滑动窗口大小
cnn_winWord_length = embedding_length * window
cnn_pos_length = 10   #位置表的embedding长度
cnn_length = cnn_winWord_length + 2 * cnn_pos_length

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

    train_rnn_data = np.load("../rcatt_train_rnn_data.npy")
    train_cnn_data = np.load("../rcatt_train_cnn_data.npy")
    train_label = np.load("../rcatt_train_label.npy")
    return train_rnn_data,train_cnn_data,train_label

def read_test_data():

    test_rnn_data = np.load("../rcatt_test_rnn_data.npy")
    test_cnn_data = np.load("../rcatt_test_cnn_data.npy")
    test_label = np.load("../rcatt_test_label.npy")
    return test_rnn_data,test_cnn_data,test_label

def attenRCnn(train_lexical_data,train_sentence_data,train_label,
         test_lexical_data,test_sentence_data,test_label):

    train_label = np_utils.to_categorical(train_label,label_count)
    test_label_categorical =  np_utils.to_categorical(test_label,label_count)

    rnn_branch = Sequential()
    rnn_branch.add(Bidirectional(SimpleRNN(500,return_sequences=True),merge_mode='concat',
                      input_shape=(rnn_step,embedding_length)))
    rnn_branch.add(Activation('tanh'))
    rnn_branch.add(GlobalMaxPooling1D())
    rnn_branch.add(Dense(200,activation='tanh'))

    cnn_branch = Sequential()
    cnn_branch.add(Convolution1D(500,3,border_mode='same',input_shape
                =(timestep,cnn_length),W_regularizer=l2(0.01)))
    cnn_branch.add(GlobalMaxPooling1D())
    cnn_branch.add(Dropout(0.6))
    cnn_branch.add(Dense(200,activation='tanh',W_regularizer=l2(0.01)))

    merged = Merge([rnn_branch,cnn_branch],mode='concat')
    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dense(50,activation='tanh',W_regularizer=l2(0.01)))
    final_model.add(Dense(label_count,activation='softmax',name='output'))


    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    final_model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

    epoch = 50
    for i in range(epoch):
        print i,'/',epoch
        final_model.fit([train_lexical_data, train_sentence_data],train_label, batch_size=64,
                        nb_epoch=1, shuffle=True,verbose=1,validation_data=
                        ([test_lexical_data,test_sentence_data],test_label_categorical))
        # 带权重的类别
        # model.fit([train_lexical_data, train_sentence_data],train_label, batch_size=64,
        #           nb_epoch=10, class_weight=classWeight, shuffle=True,verbose=1)
        # loss_and_metrics = model.evaluate([test_lexical_data,test_sentence_data], test_label_categorical, batch_size=32)
        # print loss_and_metrics

        test_classes = final_model.predict_classes([test_lexical_data,test_sentence_data],batch_size=64)
        putOut(test_classes,test_label)

    final_model.save('../final_model.h5')

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
    print '\naverage fscore: ',sum_fscore / label_count

if __name__ == "__main__":

     train_rnn_data,train_cnn_data,train_label = read_train_data()
     test_rnn_data,test_cnn_data,test_label = read_test_data()

     class_weight()

     attenRCnn(train_rnn_data,train_cnn_data,train_label,
         test_rnn_data,test_cnn_data,test_label)
