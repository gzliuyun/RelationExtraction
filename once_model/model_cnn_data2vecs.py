#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import gensim
import numpy as np
from database_handle import LTP

from keras.models import Sequential
from keras.layers import Embedding

rel2int = {}
position_dict = {}

timestep = 80    #每条语料设定的最大长度
embedding_length = 100   #embedding的维度
lexical_length = 20     #词汇级别的单词数   4*5, 最多4个人名,每个人名包括前后共5个词
pos_embedding_length = 10   #位置表的的embedding长度
#句子级别的embedding总长度
sentence_embedding_length = embedding_length + 2 * pos_embedding_length

word2vec = gensim.models.Word2Vec.load('../word2vec/wiki.zh.text.vector')
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
def sen2list(sentence,name1,name2):
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
    if item in word2vec:
        rlist = np.array(map(float, word2vec[item]))
        return rlist
    # 如果 item 不在训练好的model词汇中,则随机出一个embedding
    else:
        rlist = map(float, list(4*np.random.rand(embedding_length)-2))   #-2到2直接
        return np.array(rlist)

def lexical2embedding(wordList,name1,name2):
    lexicalEmbed = np.empty((lexical_length,embedding_length),dtype="float64")
    length = len(wordList)
    def addLexicalEmbed(label,index):
        if index - 2 >= 0:
            lexicalEmbed[label] = word2embedding(wordList[index-2])
            label += 1
        else:
            lexicalEmbed[label] = np.array([0] * embedding_length)
            label += 1

        ##########################
        if index - 1 >= 0:
            lexicalEmbed[label] = word2embedding(wordList[index-1])
            label += 1
        else:
            lexicalEmbed[label] = np.array([0] * embedding_length)
            label += 1

        ##########################
        lexicalEmbed[label] = word2embedding(wordList[index])
        label += 1

        ##########################
        if index + 1 < length:
            lexicalEmbed[label] = word2embedding(wordList[index+1])
            label += 1
        else:
            lexicalEmbed[label] = np.array([0] * embedding_length)
            label += 1

        ##########################
        if index + 2 < length:
            lexicalEmbed[label] = word2embedding(wordList[index+2])
            label += 1
        else:
            lexicalEmbed[label] = np.array([0] * embedding_length)
            label += 1
        return label

    label = 0
    for index in range(length):
        word = wordList[index]
        if word == name1 or word == name2:
            label = addLexicalEmbed(label,index)
            if label >= lexical_length: break
    while(label < lexical_length):
        lexicalEmbed[label] = np.array([0] * embedding_length)
        label += 1

    return lexicalEmbed

def sentence2embedding(wordList,name1,name2):
    sentenceEmbedding = np.empty((timestep,sentence_embedding_length),dtype="float64")
    posName1 = []
    posName2 = []

    for index in range(len(wordList)):
        if wordList[index] == name1:
            posName1.append(index)
        elif wordList[index] == name2:
            posName2.append(index)

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

    global position_dict
    index = 0
    for index in range(len(wordList)):
        if index >= timestep:
            break
        word = wordList[index]
        ebd = []
        ebd.extend(word2embedding(word))
        pos1,pos2 = minDistence(index)
        if not position_dict.has_key(pos1):
            position_dict[pos1] = list( 4 * np.random.rand(pos_embedding_length) - 2 )
        if not position_dict.has_key(pos2):
            position_dict[pos2] = list( 4 * np.random.rand(pos_embedding_length) - 2)
        ebd.extend(position_dict[pos1])
        ebd.extend(position_dict[pos2])

        sentenceEmbedding[index] = np.array(ebd)

    index += 1
    while index < timestep:
        ebd = [0] * sentence_embedding_length
        sentenceEmbedding[index] = np.array(ebd)
        index += 1

    return sentenceEmbedding

def corpus_embedding(filePath,length):
    file = open(filePath)
    lexical_data = np.empty((length,lexical_length,embedding_length),dtype="float64")
    sentence_data = np.empty((length,timestep,sentence_embedding_length),dtype="float64")
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
        wordsList = sen2list(sentence,name1,name2)
        lexical_data[index] = lexical2embedding(wordsList,name1,name2)
        sentence_data[index]  = sentence2embedding(wordsList,name1,name2)
        data_label[index] = rel2int[relation]
        index += 1
    return lexical_data,sentence_data,data_label

def store_embedding():
    trainFilePath = '../trainSentences.txt'
    testFilePath = '../testSentences.txt'

    train_lexical_data,train_sentence_data,train_label = corpus_embedding(trainFilePath,4136)
    np.save("../cnn_train_lexical_embedding_data.npy",train_lexical_data)
    np.save("../cnn_train_sentence_embedding_data.npy",train_sentence_data)
    np.save("../cnn_train_embedding_label.npy",train_label)

    test_lexical_data,test_sentence_data,test_label = corpus_embedding(testFilePath,1037)
    np.save("../cnn_test_lexical_embedding_data.npy",test_lexical_data)
    np.save("../cnn_test_sentence_embedding_data.npy",test_sentence_data)
    np.save("../cnn_test_embedding_label.npy",test_label)

if __name__ == "__main__":
    relation2int()
    store_embedding()