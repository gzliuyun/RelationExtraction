#__author__ = 'Administrator'
# -*- coding: utf-8 -*-
# from database_handle import LTP

if __name__ == "__main__":
    fr = open('../people.list_wiki.zh.text.jian')
    fw = open('../people.list_wiki.zh.text.jian_seg','w+')
    for line in fr:
        print line.strip()
        break
        sents = LTP.sentence_splitter(line.strip())
        first = True
        for sentence in sents:
            words = LTP.cut_words(sentence)
            for word in words:
                if len(word.strip()) == 0:
                    continue
                if first:
                    fw.write(word)
                    first = False
                else:
                    fw.write(' ')
                    fw.write(word)
        fw.write('\n')
    fr.close()
    fw.close()
