#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
from database_handle.mysql_handle import  CMySql
from database_handle import LTP
keyWordsCount = 20 #每种关系选择的特征个数
topkeyWords = {}   #每种关系包含的特征词列表
wordsCountAll = {}  #每个单词在所有类别中出现的次数
wordsCountInRel = {}  #每个单词在每个分类关系中出现的次数
db = CMySql()
db.setoptions('192.168.1.11',3306,'root','root','RelationExtraction')


def statistics(sentence,rel):
    if isinstance(sentence,unicode):
        sentence = sentence.encode('utf-8')

    print sentence
    # 分词
    words = LTP.cut_words(sentence)
    #词性标注
    tags = LTP.post_tagger(words)
    wordsSet = set()
    for i in range(len(words)):
        if tags[i]!= 'n' and tags[i]!= 'v':
            continue
        word = words[i].strip()
        # 去重
        if word == '编辑':  #语料中包含大量'编辑'字符串，去掉
            continue
        if (len(unicode(word,'utf-8')) <= 1) or (word in wordsSet):
            continue

        wordsSet.add(word)

        if len(word) <= 1:
            continue
        if not wordsCountInRel[rel].has_key(word):
            wordsCountInRel[rel][word] = 1
        else:
            wordsCountInRel[rel][word] += 1

        if not wordsCountAll.has_key(word):
            wordsCountAll[word] = 1
        else:
            wordsCountAll[word] += 1

def topValueWords(rel):

    rdict = {}
    for word in wordsCountInRel[rel].keys():
        rdict[word] = 1.0 *wordsCountInRel[rel][word] * wordsCountInRel[rel][word] / wordsCountAll[word]
    lst = sorted(rdict.items(),key= lambda item:item[1],reverse=True)
    rlst = []
    for i in range(keyWordsCount):
        key = lst[i][0]
        value = lst[i][1]
        rlst.append(key)
    return rlst

def sentenceScore(id,sentence,name1,name2,rel):
    if isinstance(id,unicode):
        id = id.encode('utf-8')
    if isinstance(sentence,unicode):
        sentence = sentence.encode('utf-8')
    if isinstance(name1,unicode):
        name1 = name1.encode('utf-8')
    if isinstance(name2,unicode):
        name2 = name2.encode('utf-8')

    words = LTP.cut_words(sentence)
    #词性标注
    tags = LTP.post_tagger(words)
    # 命名实体识别
    netags = LTP.ner(words,tags)

    lenSentence = len(words)  #句子中包含的单词个数
    keyWordsInSentence = 0 #句子中出现的关键词个数 N1
    otherNameCount = 0   #句子中包含的除了name以外的其他人名数 N2
    punctuationBetweenName12 = 0 # name1和name2之间间隔的标点符号数量 N3
    wordsCountBetweenName12 = 0  # name1和name2之间间隔的词汇数量 S1

    name1Index = 0   #name1的下标
    name2Index = 0   #name2的下标
    nameStr =""
    nameSet = set()   #句子中包含的除了name以外的其他人名
    for index in range(len(netags)):
        ng = netags[index]
        if ng == 'S-Nh' or ng == 'E-Nh':
            nameStr += words[index]
            if nameStr != name1 or nameStr != name2 :
                nameSet.add(nameStr)
            elif nameStr == name1:
                name1Index = index
            elif nameStr == name2:
                name2Index == index
            nameStr = ""
        elif ng == 'B-Nh' or ng == 'I-Nh':
            nameStr += words[index]

    otherNameCount = len(nameSet)
    wordsCountBetweenName12 =  abs(name2Index - name1Index)
    for index in range( min(name1Index,name2Index), max(name1Index,name2Index) ):
        if tags[index] == 'wp':
            punctuationBetweenName12 += 1
    print 'keyWords:    ',
    for word in set(words):
        if word in topkeyWords[rel]:
            print word,
            keyWordsInSentence += 1
    print
    score = 1.0 * (100 * keyWordsInSentence - 20 * otherNameCount - 10 * punctuationBetweenName12 - 3 * wordsCountBetweenName12)/lenSentence

    print 'lenSentence: ', lenSentence
    print 'keyWordsInSentence:  ', keyWordsInSentence
    print 'otherNameCount:  ', otherNameCount
    print 'punctuationBetweenName12:    ', punctuationBetweenName12
    print 'wordsCountBetweenName12: ', wordsCountBetweenName12

    sql = "update rel_in_sentence set relation = '%s' ,score = %s where id = '%s' " %(rel,score,id)
    db.uptable(sql)
    print sentence
    print rel,name1,name2,score
    print '----------------------------------------'

if __name__ == "__main__":
    relation_list = ['man_and_wife','parent_children','teacher_and_pupil','cooperate','friends','brother','sweetheart']
    # relation_list = ['brother','sweetheart']
    relation_dict = {}
    relation_dict['man_and_wife'] = [
        '妻子','丈夫','老公','老婆','夫妻','夫人','太太'
        '妻','夫'
    ]
    relation_dict['parent_children'] = [
        '父亲','爸爸','母亲','妈妈','女儿','父母','子女','长女','次女',
        '母子','父女','儿子','长子','次子','子','三儿子','四子'
        '父','爸','爹','母','妈','娘','儿','女'
    ]
    relation_dict['teacher_and_pupil'] = [
        '老师','师父','师傅','恩师','导师','弟子','学生','徒弟',
        '师','徒'
    ]
    relation_dict['cooperate'] = [
        '搭档','合作','同事','同僚','队友','女双搭档','小品搭档','男双搭档','沙排搭档',
        '羽毛球混双搭档','羽毛球女双搭档','队友及搭档','双人滑搭档'
    ]
    relation_dict['friends'] = [
        '好友','朋友','挚友','友人','密友'
        '友'
    ]
    relation_dict['brother'] = [
        '哥哥','弟弟','大哥','兄弟','兄长','兄','长兄','三哥','三弟','二哥'
        '哥','弟'
    ]
    relation_dict['sweetheart'] = [
        '前女友','前男友','绯闻男友','男友','女友','恋人',
        '男朋友','初恋男友','初恋'
    ]

    for rel in relation_list:
        print rel
        wordsCountInRel[rel] = {}
        for keyWords in relation_dict[rel]:
            if isinstance(keyWords,unicode):
                keyWords = keyWords.encode('utf-8')
            sql = "select sentence from rel_in_sentence where type = '%s'" % keyWords
            result = db.query(sql)
            for row in result:
                statistics(row[0],rel)
    print '--------------------------------'
    # 将关键词写入文本
    f = open('../keyWords.txt','w+')
    for rel in relation_list:
        topkeyWords[rel] = topValueWords(rel)
        #将细分的关系也加入关键词集合中
        for item in relation_dict[rel]:
            if item not in topkeyWords[rel]:
                topkeyWords[rel].append(item)

        f.write(rel)
        f.write('\n')
        for index in range( len(topkeyWords[rel]) ):
            kw = topkeyWords[rel][index]
            if index == 0:
                f.write(kw)
            else:
                f.write('   ')
                f.write(kw)
        f.write('\n')
    f.close()

    for rel in relation_list:
        for keyWords in relation_dict[rel]:
            if isinstance(keyWords,unicode):
                keyWords = keyWords.encode('utf-8')
            sql = "select id,sentence,name1,name2 from rel_in_sentence where type = '%s'" % keyWords
            result = db.query(sql)
            for row in result:
                id = row[0]
                sentence = row[1]
                name1 = row[2]
                name2 = row[3]
                sentenceScore(id,sentence,name1,name2,rel)
