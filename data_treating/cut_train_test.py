#__author__ = 'Administrator'
# -*- coding: utf-8 -*-

def train_test():
    sourceFile = open('../relInSen.txt','r')
    # trainFile = open('../trainSentences.txt','w+')
    # testFile = open('../testSentences.txt','w+')

    rel2num = {}   #关系对应的总数量
    testPercent = 0.2  #   测试数据占总数理的比重
    rel2testnum = {}  #关系对应在测试数据中比列计算的数量
    trainNumber ={}  #训练数据统计数量
    testNumber = {} #测试数据统计数量
    # 统计每种关系包含的数量
    sumSource = 0   #原数据总量
    sumTrain = 0  #训练数据总量
    sumTest = 0  #测试数据总量
    for line in sourceFile:
        sumSource += 1
        if isinstance(line,unicode):
            line = line.encode('utf-8')
        lineSplit = line.strip().replace(' ','').split('#')
        # sentence = line[0]
        # name1 = line[1]
        # name2 = line[2]
        relation = lineSplit[3]
        if relation in rel2num:
            rel2num[relation] += 1
        else:
            rel2num[relation] = 1
    #     计算测试数据的数量
    print 'SourceFile number :' ,sumSource
    for relation,num in rel2num.items():
        print relation,num
        rel2testnum[relation] = rel2num[relation] * testPercent
        rel2num[relation] = 0
        trainNumber[relation] = 0
        testNumber[relation] = 0
    print '--------------------'
    sourceFile.seek(0) #文件指针回到文件开头
    for line in sourceFile:
        if isinstance(line,unicode):
            line = line.encode('utf-8')
        lineSplit = line.strip().replace(' ','').split('#')
        # sentence = line[0]
        # name1 = line[1]
        # name2 = line[2]
        relation = lineSplit[3]
        # 隔三条抽取一条测试数据
        if ( rel2testnum[relation] > 0 ) and ( rel2num[relation] % 3 == 0) :
            # testFile.write(line)
            rel2num[relation] += 1
            rel2testnum[relation] -= 1
            testNumber[relation]  += 1
            sumTest += 1
        else:
            # trainFile.write(line)
            rel2num[relation] += 1
            trainNumber[relation] += 1
            sumTrain += 1
    print 'TrainFile numer:', sumTrain
    for relation, num in trainNumber.items():
        print relation,num
    print '--------------------'
    print 'TestFile number:',sumTest
    for relation, num in testNumber.items():
        print relation,num

    sourceFile.close()
    # trainFile.close()
    # testFile.close()

if __name__ =="__main__":

    train_test()