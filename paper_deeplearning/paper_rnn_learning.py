#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM,Embedding,Flatten,SimpleRNN
from keras.utils import np_utils
from keras.layers.convolutional import  MaxPooling2D,MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.layers.wrappers import Bidirectional
from keras.preprocessing import  sequence
from keras.layers.core import Dropout,Lambda

# 导入sklear的数据shuffle
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle

timestep = 80    #每条语料设定的最大长度
embedding_length = 100   #embedding的维度
label_count = 7
classWeight = {}   #类别权重，处理类别数据不平衡用

def read_embedding():
    train_data = np.load("../rnn_train_embedding_data.npy")
    train_label = np.load("../rnn_train_embedding_label.npy")

    test_data = np.load("../rnn_test_embedding_data.npy")
    test_label = np.load("../rnn_test_embedding_label.npy")

    return train_data,train_label,test_data,test_label

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

def rnn(train_data, train_label,test_data,test_label):
    train_label = np_utils.to_categorical(train_label,label_count)
    test_label_categorical = np_utils.to_categorical(test_label,label_count)

    model = Sequential()
    model.add(Bidirectional(SimpleRNN(600,return_sequences=True),merge_mode='concat',input_shape=(timestep,embedding_length)))
    model.add(Activation('tanh'))

    model.add(GlobalMaxPooling1D())
    model.add(Dense(200,activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(label_count,activation='softmax'))

    #使用SGD + momentum
    #model.compile里的参数loss就是损失函数(目标函数),decay为衰减率
    #调用fit方法，就是一个训练过程. 训练的epoch数设为10，batch_size为100．
    #数据经过随机打乱shuffle=True。verbose=1，训练过程中输出的信息，0、1、2三种方式都可以，无关紧要。show_accuracy=True，训练时每一个epoch都输出accuracy。
    #validation_split=0.2，将20%的数据作为验证集。
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    epoch = 30
    for i in range(epoch):
        print i,'/', epoch
        model.fit(train_data, train_label, batch_size=128, nb_epoch=1, shuffle=True,
              verbose=1, validation_data=(test_data, test_label_categorical))
        test_classes = model.predict_classes(test_data,batch_size=32)
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

    print 'average fscore: ',sum_fscore/label_count

if __name__ == "__main__":
    train_data, train_label, test_data, test_label = read_embedding()
    class_weight()
    rnn(train_data,train_label,test_data,test_label)

