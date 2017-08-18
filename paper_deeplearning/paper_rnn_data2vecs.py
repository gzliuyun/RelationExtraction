#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  #加入当前文件路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))   #加入上级本舰路径
import gensim
import numpy as np
from database_handle import LTP
from keras.models import Sequential
from keras.layers import Embedding

rel2int = {}
#word embedding 的维度

#位置标签（<e1></e1><e2></e2>）下标对应的embedding
position_tag_embedding_array = []
position_tag =['<e1>','</e1>','<e2>','</e2>']

timestep = 80    #每条语料设定的最大长度
embedding_length = 100   #embedding的维度

model = gensim.models.Word2Vec.load_word2vec_format('../word2vec/wiki.zh.text.vector')
def relation2int():
    global rel2int
    rel2int = {
        'man_and_wife': 0,
        'parent_children': 1,
        'teacher_and_pupil': 2,
        'cooperate': 3,
        'friends': 4,
        'brother': 5,
        'sweetheart': 6
    }

# 利用keras的embedding功能将位置标签<e1>,</e1>,<e2>,</e2>用embedding_length维的embedding表示
def position_tag_embeddings():
    model = Sequential()
    model.add(Embedding(4,embedding_length,input_length=1))
    model.compile('rmsprop','mse')

    global position_tag_embedding_array   #标签列表中位置下标对应标签的embedding表示
    position_tag_embedding_array = np.array([0,1,2,3])
    position_tag_embedding_array = model.predict(position_tag_embedding_array)

def sen2embed(wordsList):
    # 获取item 单词对应的embedding
    def word2embedding(item):
        if not isinstance(item,unicode):
            item = unicode(item,'utf-8')
        if item in model:
            rlist = np.array(map(float, model[item]))
            return rlist
        # 如果 item 不在训练好的model词汇中,则随机出一个embedding
        else:
            rlist = map(float, list(4*np.random.rand(embedding_length)-2))
            return np.array(rlist)

     # 获取标签对应的提前训练好的embedding
    def tag2embedding(item):
        index = position_tag.index(item)
        rlist = map(float,position_tag_embedding_array[index][0])
        return rlist
    sentence_embedding = np.empty((timestep,embedding_length),dtype="float64")
    index = 0
    for word in wordsList:
        if word in position_tag:
            sentence_embedding[index] = tag2embedding(word)
        else:
            sentence_embedding[index] = word2embedding(word)
        index += 1
        if index >= timestep:   break
    while index < timestep:
        sentence_embedding[index] = np.array([0] * embedding_length)
        index += 1
    return sentence_embedding

def names_tag(sentence,name1,name2):
    words = LTP.cut_words(sentence)
    words_list = []

    index = 0
    while (index < len(words)):
        word = words[index]
        if word == name1:
            words_list.append('<e1>')
            words_list.append(name1)
            words_list.append('</e1>')
            index += 1
            continue
        elif word == name2:
            words_list.append('<e2>')
            words_list.append(name2)
            words_list.append('</e2>')
            index += 1
            continue
        copy_index = index
        find = False
        while name1.find(word) == 0:
            if name1 == word:
                words_list.append('<e1>')
                words_list.append(name1)
                words_list.append('</e1>')
                find = True
                break
            copy_index += 1
            if copy_index >= len(words):
                break
            word += words[copy_index]
        if find:
            index = copy_index +1
            continue

        word = words[index]
        copy_index = index
        while name2.find(word) == 0:
            if name2 == word:
                words_list.append('<e2>')
                words_list.append(name2)
                words_list.append('</e2>')
                find = True
                break
            copy_index += 1
            if copy_index >= len(words):
                break
            word += words[copy_index]
        if find:
            index = copy_index + 1
            continue

        word = words[index]
        words_list.append(word)
        index += 1
    return words_list

def corpus_embedding(filePath,length):
    file = open(filePath)
    data_embedding = np.empty((length,timestep,embedding_length),dtype="float64")
    data_label = np.empty((length,1),dtype="uint8")
    index = 0
    for line in file:
        print index
        line = line.strip().replace(' ','').split('#')
        if isinstance(line,unicode):
            line = line.encode('utf-8')
        sentence = line[0]
        name1 = line[1]
        name2 = line[2]
        relation = line[3]
        wordsList = names_tag(sentence,name1,name2)
        data_embedding[index] = sen2embed(wordsList)
        # for i in range(len(data_embedding[index])):
        #     print data_embedding[index][i]
        data_label[index] = rel2int[relation]
        index += 1
    return data_embedding,data_label

def store_embedding():
    trainFilePath = '../trainSentences.txt'
    testFilePath = '../testSentences.txt'

    train_data,train_label = corpus_embedding(trainFilePath,4136)
    np.save("../rnn_train_embedding_data.npy",train_data)
    np.save("../rnn_train_embedding_label.npy",train_label)

    test_data,test_label = corpus_embedding(testFilePath,1037)
    np.save("../rnn_test_embedding_data.npy",test_data)
    np.save("../rnn_test_embedding_label.npy",test_label)

if __name__ == "__main__":
    relation2int()
    position_tag_embeddings()
    store_embedding()