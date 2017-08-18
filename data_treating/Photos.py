#__author__ = 'Administrator'
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import urllib,urllib2
from lxml import etree
import os
from database_handle.mysql_handle import CMySql
import requests
from qiniu import Auth, put_file, etag, urlsafe_base64_encode
import qiniu.config
# 数据库连接
db = CMySql()
db.setoptions('192.168.1.11',3306,'root','root','RelationExtraction')

#七牛云配置
#需要填写你的 Access Key 和 Secret Key
access_key = 'GLCCcNfZiqOx4pjL6SDZGRfDu-QMX6cYMB9Yupku'
secret_key = 'TOPsxBISmd091aVza3WNdiJwTPRtgHY_k8aWBOdj'
#构建鉴权对象
q = Auth(access_key, secret_key)
#要上传的空间
bucket_name = 'relations'

# 图片爬虫
def craw(url,id):
    try:
        request = urllib2.Request(url)
        response = urllib2.urlopen(request,timeout=10)
        html = response.read()

        soup = BeautifulSoup(html,"lxml")
        div = soup.select(".doc-img")[0]
        img = div.contents[1].contents[0]
        imgUrl = img.get("src")

        print imgUrl
        urllib.urlretrieve(imgUrl,'../photos/%s.jpg' % id)
        return True
    except:
        print "picture error ........"
        return False

# 图片上传七牛云，向数据库插入图片地址
def upload_qiniu(id):
    # 上传到七牛后保存的文件名
    key = '%s.jps' % id
    #生成上传Token,可以指定过期时间等
    token = q.upload_token(bucket_name,key,3600)
    #要上传的本地文件路径
    localPhoto = '../photos/%s.jpg' % id
    ret,info = put_file(token,key,localPhoto)

    photo_url = 'http://or7kybwo9.bkt.clouddn.com/%s.jps' % id
    sql = "update people_list set photo_url = '%s' where id = %s " % (photo_url,id)
    db.insert(sql)

if __name__ == "__main__":
    sql = "select id,url from people_list where url is not NULL and photo_url is NULL and id > 5300"
    result = db.query(sql)
    print len(result)
    for items in result:
        id = items[0]
        url = items[1]
        print id
        if url == None: continue
        url = url.encode("utf-8")
        if craw(url,id):
            upload_qiniu(id)
    print 'finished ......'

