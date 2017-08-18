#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Sequential,Model,load_model
from keras.utils import np_utils

def read_test_data():
    test_rnn_data = np.load("../rcatt_test_rnn_data.npy")
    test_cnn_data = np.load("../rcatt_test_cnn_data.npy")
    test_label = np.load("../rcatt_test_label.npy")
    return test_rnn_data,test_cnn_data,test_label

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
    label_count = 7
    model = load_model('../final_model.h5')
    test_rnn_data, test_cnn_data,test_label = read_test_data()
    # test_label_categorical =  np_utils.to_categorical(test_label,label_count)
    print test_rnn_data.shape
    print test_cnn_data.shape
    test_classes = model.predict_classes([test_rnn_data,test_cnn_data],batch_size=1)
    for cls in test_classes:
        print cls
    # putOut(test_classes,test_label)