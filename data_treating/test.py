#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
# import gensim
# model = gensim.models.Word2Vec.load_word2vec_format('../wiki.zh.text.vector')
# result = model.most_similar(u"男人")
# print model[u'男人']
# # for e in result:
# # 	if isinstance(e[0],unicode):
# # 		tp = e[0].encode('utf-8')
# # 		print tp
# # print model[u'习近平']
# # print model[u'毛泽东']
# if u'毛泽东' in model:
#     print '毛yes'
#
# if not u'刘赟的' in model:
#     print 'no'
import os
print os.path.dirname(os.path.abspath(__file__))

print os.path.abspath(os.path.join(os.path.dirname(__file__),".."))

classWeight = {}

def class_weight():
    # 'man_and_wife': 0,
    # 'parent_children': 1,
    # 'teacher_and_pupil': 2,
    # 'cooperate': 3,
    # 'friends': 4,
    # 'brother': 5,
    # 'sweetheart': 6
    global classWeight
    relationsNumber = [1395,1451,1030,537,207,225,328]
    weight = [1.0*number/100 for number in relationsNumber]
    multiplySum = 1.0
    for w in weight:
        multiplySum *= w
    for i in range(len(weight)):
        weight[i] = multiplySum / weight[i]
    minWeight = min(weight)
    dec = 1
    while minWeight > 1:
        dec *= 10
        minWeight /= 10
    for i in range(len(weight)):
        # 小数点保留三位
        weight[i] = round(weight[i] / dec,3)
        classWeight[i] = weight[i]

if __name__ =="__main__":
    class_weight()