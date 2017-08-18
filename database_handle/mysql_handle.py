#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import os.path
try:
    import MySQLdb
except ImportError:
    raise ImportError("[E]: MySQLdb module not found!")

class CMySql(object):
    def __init__(self):
        self.Option = {"host" : "",
                       "port" : "",
                       "username" : "",
                       "password" : "",
                       "database" : ""}

    def setoptions(self, host, port, user, pwd, db):
        self.Option["host"] = host
        self.Option["port"]=port
        self.Option["username"] = user
        self.Option["password"] = pwd
        self.Option["database"] = db
        self.start()

    def start(self):
        try:
            self.db = MySQLdb.connect(
                        host = self.Option["host"],
                        user = self.Option["username"],
                        passwd = self.Option["password"],
                        db = self.Option["database"],
                        port = self.Option["port"],
                        charset="utf8"
            )
        except :
            raise Exception("[E] Cannot connect to %s" % self.Option["host"])

    def insert(self, sqlstate,param=()):
        """
        @todo: 虽然函数名是insert，不过增删改都行
        """
        self.cursor = self.db.cursor()
        self.cursor.execute(sqlstate,param)
        self.db.commit()

    def uptable(self,sql):   #更新table
        self.cursor = self.db.cursor()
        self.cursor.execute(sql)
        self.db.commit()

    def query(self, sqlstate,param=()):
        self.cursor = self.db.cursor()
        self.cursor.execute(sqlstate,param) #查
        qres = self.cursor.fetchall()
        return qres

    def one_query(self, sqlstate,param=()):
        self.cursor = self.db.cursor()
        self.cursor.execute(sqlstate,param) #查
        qres = self.cursor.fetchall()[0]
        return qres

    def close(self):
        self.db.close()
