#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
from database_handle.mysql_handle import CMySql
from database_handle import LTP

db = CMySql()
db.setoptions('192.168.1.11',3306,'root','root','RelationExtraction')

def name2id(name):
    try:
        sql = "select id from people_list where name = '%s'" %(name)
        result = db.query(sql)
    except:
        return -1
    if len(result) == 0:
        return -1
    name_id = result[0][0]
    if not name_id is None:
        return name_id
    else:
        return -1

def compare_with_database(name1_id,name2_id):
    try:
        sql = "select relation_id from people_relation where name1_id = '%s' and name2_id = '%s'" \
            %(name1_id, name2_id)
        result = db.query(sql)
    except:
        return -1
    if len(result) == 0:
        return -1
    relation_id = result[0][0]
    if not relation_id is None:
        sql =  "select type from relation_list where id = '%s'" % (relation_id)
        result = db.query(sql)
        type = result[0][0]
        if isinstance(type,unicode):
            type = type.encode('utf-8')
        return type
    else:
        return -1

def add_rel_in_sentence(sentence,name1,name2,relation_type,url):
    try:
        sql = "select max(id) from rel_in_sentence"
        result = db.query(sql)
    except:
        return
    sentence_id = result[0][0]
    if sentence_id is None:
        sentence_id = 0
    sentence = sentence.replace("'","''").strip()
    try:
        sql = "insert ignore into rel_in_sentence(id,sentence,name1,name2,url,type) values('%s','%s','%s','%s','%s','%s')"\
            %(sentence_id + 1, sentence,name1,name2,url,relation_type)
        db.insert(sql)
    except:
        return

# 未标记的人名对句子
def add_no_type_rel(sentence,name1,name2,url):
    try:
        sql = "select max(id) from no_type_rel"
        result = db.query(sql)
    except:
        return
    sentence_id = result[0][0]
    if sentence_id is None:
        sentence_id = 0
    sentence = sentence.replace("'","''").strip()
    try:
        sql = "insert ignore into no_type_rel(id,sentence,name1,name2,url) values('%s','%s','%s','%s','%s')"\
            %(sentence_id + 1, sentence,name1,name2,url)
        db.insert(sql)
    except:
        return

def person_information(mainName,introduction,url):
    print mainName
    mainName_id = name2id(mainName)
    sentences = LTP.sentence_splitter(introduction)
    for sen in sentences:
        if len(sen.strip()) == 0:
            continue
        # 分词
        words = LTP.cut_words(sen)
        # print '\n'.join(words)

        if sen.find(mainName) != -1:
            #词性标注
            tags = LTP.post_tagger(words)
            # 命名实体识别
            netags = LTP.ner(words,tags)

            nameSet = set()   #句子中包含的除了name以外的其他人名
            nameStr =""
            for index in range(len(netags)):
                ng = netags[index]
                if ng == 'S-Nh' or ng == 'E-Nh':
                    nameStr += words[index]
                    if not nameStr == mainName:
                        nameSet.add(nameStr)
                    nameStr = ""
                elif ng == 'B-Nh' or ng == 'I-Nh':
                    nameStr += words[index]

            for otherName in nameSet:
                otherName_id = name2id(otherName)
                if mainName_id != -1 and otherName_id != -1:
                    relation_type = compare_with_database(mainName_id,otherName_id)
                    if relation_type != -1:
                        print sen
                        add_rel_in_sentence(sen,mainName,otherName,relation_type,url)
                    else:
                        add_no_type_rel(sen,mainName,otherName,url)


if __name__ =="__main__":
    sql = 'select name,introduction,url from people_list where id > 26373 and introduction is not NULL'
    result = db.query(sql)
    for index in range(len(result)):
        _name = result[index][0]
        _introduction = result[index][1]
        _url = result[index][2]

        if isinstance(_name,unicode):
            _name = _name.encode('utf-8')
        if isinstance(_introduction,unicode):
            _introduction = _introduction.encode('utf-8')
        if isinstance(_url,unicode):
            _url = _url.encode('utf-8')

        person_information(_name, _introduction, _url)


