#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import gensim
import numpy as np
from database_handle import LTP

from keras.models import Sequential
from keras.layers import Embedding

rel2int = {}
cnn_position_dict = {}

timestep = 80    #每条语料设定的最大长度
embedding_length = 100   #embedding的维度
rnn_length = embedding_length     #rnn部分的单词数
window = 3  # cnn滑动窗口大小
cnn_winWord_length = window * embedding_length
cnn_pos_length = 10   #位置表的embedding长度
cnn_length = cnn_winWord_length + 2 * cnn_pos_length

#位置标签（<e1></e1><e2></e2>）下标对应的embedding
rnn_position_tag_embedding = []
rnn_position_tag =['<e1>','</e1>','<e2>','</e2>']

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
def posTagEmbeddings():
    model = Sequential()
    model.add(Embedding(4,embedding_length,input_length=1))
    model.compile('rmsprop','mse')

    global rnn_position_tag_embedding   #标签列表中位置下标对应标签的embedding表示
    tag_array = np.array([0,1,2,3])
    rnn_position_tag_embedding = model.predict(tag_array)

def sentence2list(sentence,name1,name2):
    words = LTP.cut_words(sentence)
    words_list = []
    index = 0
    while (index < len(words)):
        word = words[index]
        if word == name1 or word == name2:
            words_list.append(word)
            index += 1
            continue

        # 连续的分词组合起来是不是等于name1
        copy_index = index
        find = False
        while name1.find(word) == 0:
            if name1 == word:
                words_list.append(name1)
                find = True
                break
            copy_index += 1
            if copy_index >= len(words):
                break
            word += words[copy_index]
        if find:
            index = copy_index +1
            continue
        # 连续的分词组合起来是不是等于name2
        word = words[index]
        copy_index = index
        while name2.find(word) == 0:
            if name2 == word:
                words_list.append(name2)
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

# 获取item 单词对应的embedding
def word2embedding(item):
    if not isinstance(item,unicode):
        item = unicode(item,'utf-8')
    if item in model:
        rlist = np.array(map(float, model[item]))
        return rlist
    # 如果 item 不在训练好的model词汇中,则随机出一个embedding
    else:
        rlist = map(float, list(4*np.random.rand(embedding_length)-2))   #-2到2直接
        return np.array(rlist)

def attentionWeight(wordsList,name1,name2):
    length = len(wordsList)
    dot1List = []
    dot2List = []
    name1Ebd = word2embedding(name1)
    name2Ebd = word2embedding(name2)
    for i in range(length):
        word = wordsList[i]
        wordEbd = word2embedding(word)
        dot1 = np.dot(name1Ebd,wordEbd)
        dot2 = np.dot(name2Ebd,wordEbd)
        dot1List.append(dot1)
        dot2List.append(dot2)

    dot1Sum = sum(dot1List)
    dot2Sum = sum(dot2List)
    att_weight = []
    for i in range(length):
        dot1List[i] = 1.0 * dot1List[i]/dot1Sum
        dot2List[i] = 1.0 * dot2List[i]/dot2Sum
        weight = 1.0 * (dot1List[i] + dot2List[i]) / 2
        att_weight.append(weight)
    return att_weight

def rnnDataEmbedding(wordsList,att_weigh,name1,name2):
    rnnEmbed = np.empty((timestep,rnn_length),dtype="float64")
    length = len(wordsList)

    index = 0
    for i in range(length):
        word = wordsList[i]
        wordEbd = word2embedding(word)
        weight =  att_weigh[i]
        if word == name1:
            rnnEmbed[index] =  rnn_position_tag_embedding[rnn_position_tag.index('<e1>')]
            index += 1
            if index >= timestep: break

            rnnEmbed[index] =  weight * wordEbd
            index += 1
            if index >= timestep: break

            rnnEmbed[index] =  rnn_position_tag_embedding[rnn_position_tag.index('</e1>')]
            index += 1
            if index >= timestep: break

        elif word == name2:
            rnnEmbed[index] =  rnn_position_tag_embedding[rnn_position_tag.index('<e2>')]
            index += 1
            if index >= timestep: break

            rnnEmbed[index] =  weight * wordEbd
            index += 1
            if index >= timestep: break

            rnnEmbed[index] =  rnn_position_tag_embedding[rnn_position_tag.index('</e2>')]
            index += 1
            if index >= timestep: break

        else:
            rnnEmbed[index] = np.array(weight * wordEbd)
            index += 1
            if index >= timestep: break

    while(index < timestep):
        rnnEmbed[index] = [0] * rnn_length
        index += 1
    return rnnEmbed

def cnnDataEmbedding(wordsList,att_weight,name1,name2):
    length = len(wordsList)
    posName1 = []
    posName2 = []
    for i in range(length):
        if wordsList[i] == name1:
            posName1.append(i)
        elif wordsList[i] == name2:
            posName2.append(i)

    def lmrEmd(i):
        ebd = []
        if (i - 1)>= 0:
            wordEbd = word2embedding(wordsList[i-1])
            weight =  att_weight[i-1]
            ebd.extend(list(weight * wordEbd))
        else:
            ebd.extend(list(np.array([0] * embedding_length)))

        wordEbd = word2embedding(wordsList[i])
        weight =  att_weight[i]
        ebd.extend(list(weight * wordEbd))

        if (i + 1) < length:
            wordEbd = word2embedding(wordsList[i+1])
            weight =  att_weight[i+1]
            ebd.extend(list(weight * wordEbd))
        else:
            ebd.extend(list(np.array([0] * embedding_length)))

        return ebd

    # 最近距离计算
    def minDistence(index):
        # 计算与name1最近距离
        i = 0
        while i < len(posName1):
            if posName1[i] > index:
                break
            i += 1
        if i == len(posName1):
            if i == 0:  pos1 = timestep  #句子中没发现name1的异常情况
            else:   pos1 = index - posName1[i-1]
        elif (i - 1 >= 0):
            if abs(index - posName1[i-1]) <= abs(posName1[i] - index):
                pos1 = index - posName1[i-1]
            else:
                pos1 = index - posName1[i]
        else:
            pos1 = index - posName1[i]

        # 计算与name2最近距离
        i = 0
        while i < len(posName2):
            if posName2[i] > index:
                break
            i += 1
        if i == len(posName2):
            if i == 0:  pos2 = timestep  #句子中没发现name2的异常情况
            else:   pos2 = index - posName2[i-1]
        elif (i - 1 >= 0):
            if abs(index - posName2[i-1]) <= abs(posName2[i] - index):
                pos2 = index - posName2[i-1]
            else:
                pos2 = index - posName2[i]
        else:
            pos2 = index - posName2[i]

        return pos1, pos2

    cnnEbd = np.empty((timestep,cnn_length),dtype="float64")
    global position_dict
    idx = 0
    for idx in range(length):
        if idx >= timestep:
            break
        ebd = []
        ebd.extend(lmrEmd(idx))

        #找离index最近的name1，计算差值
        global cnn_position_dict
        pos1, pos2 = minDistence(idx)
        if not cnn_position_dict.has_key(pos1):
            cnn_position_dict[pos1] = list( 4 * np.random.rand(cnn_pos_length) - 2 )
        if not cnn_position_dict.has_key(pos2):
            cnn_position_dict[pos2] = list( 4 * np.random.rand(cnn_pos_length) - 2)
        ebd.extend(cnn_position_dict[pos1])
        ebd.extend(cnn_position_dict[pos2])

        cnnEbd[idx] = np.array(ebd)
    while idx < timestep:
        ebd = [0] * cnn_length
        cnnEbd[idx] = np.array(ebd)
        idx += 1
    return cnnEbd

def corpus_embedding(filePath,length):
    file = open(filePath)
    rnn_data = np.empty((length,timestep,rnn_length),dtype="float64")
    cnn_data = np.empty((length,timestep,cnn_length),dtype="float64")
    label = np.empty((length,1),dtype="uint8")
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
        wordsList = sentence2list(sentence,name1,name2)
        att_weight = attentionWeight(wordsList,name1,name2)  #注意力权重
        rnn_data[index] = rnnDataEmbedding(wordsList,att_weight,name1,name2)
        cnn_data[index] = cnnDataEmbedding(wordsList,att_weight,name1,name2)
        label[index] = rel2int[relation]
        index += 1
    return rnn_data,cnn_data,label

def store_embedding():
    trainFilePath = '../trainSentences.txt'
    testFilePath = '../testSentences.txt'

    train_rnn_data,train_cnn_data,train_label = corpus_embedding(trainFilePath,4136)
    np.save("../attention_train_rnn_data.npy",train_rnn_data)
    np.save("../attention_train_cnn_data.npy",train_cnn_data)
    np.save("../attention_train_label.npy",train_label)

    test_rnn_data,test_cnn_data,test_label = corpus_embedding(testFilePath,1037)
    np.save("../attention_test_rnn_data.npy",test_rnn_data)
    np.save("../attention_test_cnn_data.npy",test_cnn_data)
    np.save("../attention_test_label.npy",test_label)

if __name__ == "__main__":
    relation2int()
    posTagEmbeddings()
    store_embedding()