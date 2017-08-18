#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import Queue
import urllib2
from bs4 import BeautifulSoup
from lxml import etree
from database_handle.mysql_handle import CMySql

crawl_queue = Queue.Queue(maxsize = 0)

base_url = "http://www.baike.com/wiki/"
tail_url = "&prd=button_doc_entry"
db = CMySql()
db.setoptions('192.168.1.11',3306,'root','root','RelationExtraction')

appearNameSet = set()   #所有出现过的人名
crawledNameSet = set()     #爬取过主页的人名
appearRelSet = set()       #出现过的关系名称
nameInQueue = set()     #当前队列中存在的元素

def name2id(nameStr):
    if not nameStr in appearNameSet:
        sql = "select max(id) from people_list"
        result = db.query(sql)
        name_id = result[0][0]
        if name_id is None:
            name_id = 0

        sql = "insert ignore into people_list(id,name) values('%s','%s')" %(name_id + 1, nameStr)
        db.insert(sql)
        appearNameSet.add(nameStr)
        return name_id + 1
    else:
        sql = "select id from people_list where name = '%s'" %(nameStr)
        result = db.query(sql)
        name_id = result[0][0]
        return name_id

def rel2id(rel):
    if not rel in appearRelSet:
        sql = "select max(id) from relation_list"
        result = db.query(sql)
        rel_id = result[0][0]
        if rel_id is None:
            rel_id = 0

        sql = "insert ignore into relation_list(id,type) values('%s','%s')" %(rel_id + 1, rel)
        db.insert(sql)
        appearRelSet.add(rel)
        return rel_id + 1
    else:
        sql = "select id from relation_list where type = '%s'" %(rel)
        result = db.query(sql)
        rel_id = result[0][0]
        return rel_id

def db_updatePerson(name_id,introduction,url):
    sql = "update people_list set introduction = '%s', url = '%s' where id = '%s'" %(introduction, url, name_id)
    db.uptable(sql)

def db_addPeoleRelation(name1_id,name2_id,rel_id):

    sql = "select max(id) from people_list"
    result = db.query(sql)
    id = result[0][0]
    if id is None:
        id = 0

    sql = "insert ignore into people_relation(id,name1_id,name2_id,relation_id) values('%s','%s','%s','%s')" %(id+1,name1_id,name2_id,rel_id)
    db.insert(sql)

def name_deal(nameStr):
    index = nameStr.find('[')
    if index != -1:
        nameStr = nameStr[:index]
    return nameStr

def crawler(mainName):

    if isinstance(mainName,unicode):
        mainName = mainName.encode('utf-8')
    print mainName ,'------'

    url = str(base_url + mainName + tail_url)

    # 将mainName加入爬取过的人物集合
    crawledNameSet.add(mainName)

    try:
        request = urllib2.Request(url)
        response = urllib2.urlopen(request)
        html = etree.HTML(response.read())
    except:
        return

    try:
        mainName_id = name2id(mainName)
    except:
        print 'add  ',mainName,' error...'

    #人物介绍专栏
    content = html.xpath('//div[@id="content"]')
    if len(content) > 0:
        content_html = etree.tostring(content[0],encoding="utf-8", pretty_print=True, method="html")
        soup = BeautifulSoup(content_html)
        introduction =  soup.text.replace("\n","").replace(" ","\n").strip()
        #防止出现"'"号时引起sql错误，进行转义
        introduction = introduction.replace("'","''")
        if isinstance(introduction,unicode):
            introduction = introduction.encode('utf-8')
        try:
            db_updatePerson(mainName_id,introduction,url)
        except:
            print 'update ',mainName,' error...'

    # 人物关系栏
    relations = html.xpath('//ul[@id="fi_opposite"]')
    if len(relations) > 0:
        relations_html = etree.tostring(relations[0],encoding="utf-8", pretty_print=True, method="html")
        soup = BeautifulSoup(relations_html)
        li_list = soup.select('li')
        for name_relation in li_list:
            other_name = ""
            rel = ""
            for child in name_relation.children:
                item = child.string.strip()
                if len(item) == 0:  continue
                if other_name =="":
                    other_name = item
                else:
                    rel = item

            if isinstance(other_name,unicode):
                other_name = other_name.encode('utf-8')
            other_name = name_deal(other_name)
            if isinstance(rel,unicode):
                rel = rel.encode('utf-8')
            print other_name, rel,1

            otherName_id = name2id(other_name)
            rel_id = rel2id(rel)
            #如果other_name的主页没有爬取过，并且没有在爬取队列中，则将other_name加入爬去队列
            if (not other_name in crawledNameSet) and (not other_name in nameInQueue):
                crawl_queue.put(other_name)
                nameInQueue.add(other_name)

            if other_name != "" and rel != "":
                try:
                    db_addPeoleRelation(mainName_id,otherName_id,rel_id)
                except:
                    pass

    # 姓名栏
    holder = html.xpath('//ul[@id="holder1"]')
    if len(holder) > 0:
        relations_html = etree.tostring(holder[0],encoding="utf-8", pretty_print=True, method="html")
        soup = BeautifulSoup(relations_html)
        li_list = soup.select('li')
        for name_relation in li_list:
            other_name = ""
            rel = ""
            for child in name_relation.children:
                item = child.string.strip()
                if len(item) == 0:  continue
                if other_name =="":
                    other_name = item
                else:
                    rel = item

            if isinstance(other_name,unicode):
                other_name = other_name.encode('utf-8')
            other_name = name_deal(other_name)
            if isinstance(rel,unicode):
                rel = rel.encode('utf-8')
            print other_name, rel, 2

            otherName_id = name2id(other_name)
            rel_id = rel2id(rel)
            #如果other_name的主页没有爬取过，并且没有在爬取队列中，则将other_name加入爬去队列
            if (not other_name in crawledNameSet) and (not other_name in nameInQueue):
                crawl_queue.put(other_name)
                nameInQueue.add(other_name)

            if other_name != "" and rel != "":
                try:
                    db_addPeoleRelation(otherName_id,mainName_id,rel_id)
                except:
                    pass
def init_set():
    def addToNameSet(list):
        appearNameSet.add(list[0].encode('utf-8'))   #所有出现过的人名
        crawledNameSet.add(list[0].encode('utf-8'))     #爬取过主页的人名

    def addToRelationSet(list):
        appearRelSet.add(list[0].encode('utf-8'))       #出现过的关系名称

    sql = "select name from people_list where url is not NULL"
    nameResult = db.query(sql)
    if len(nameResult) > 0:
        map(addToNameSet, nameResult)

    sql = "select type from relation_list"
    relationResult = db.query(sql)
    if len(relationResult) > 0:
        map(addToRelationSet, relationResult)

if __name__ == "__main__":
    init_set()
    # people_list = ['习近平','马英九','李嘉诚','马化腾','成龙','范冰冰','林丹','李克强','莫言','朴槿惠','胡锦涛']
    people_list = ['令计划','周永康','王思聪','刘强东','刘华清','刘谦','郑智','贾谊','杜甫','王明']
    for item in people_list:
        if item in crawledNameSet:
            print item ,'has been crawled...'
            continue
        crawl_queue.put(item)
        nameInQueue.add(item)

    while (crawl_queue.empty() == False):
        fname = crawl_queue.get()
        nameInQueue.remove(fname)
        crawler(fname)


