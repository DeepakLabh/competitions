import pandas as pd
import numpy as np
from collections import Counter
import math
import re

class tfidf(object):
    def __init__(self, corpus_file_path):
        self.corpus_file_path = corpus_file_path
        self.corpus_list = re.split(r'\W*', open(corpus_file_path).read().lower())
        self.corpus_length = len(self.corpus_list)
	self.corpus_dict = Counter(self.corpus_list)

    def tf(self, word, sent):
        word_list = re.split(r'\W*', sent.lower())
        word_count = Counter(word_list)
        t = word_count[word]
        return t

    def idf(self, word, sent):
        try:
            #print  self.corpus_dict[word],111111,self.corpus_length,float(self.corpus_dict[word]/self.corpus_length)
            i = abs(math.log(float(self.corpus_dict[word])/self.corpus_length))
        except Exception as e:
            print e, 'could not make tfidf vector'
            i = float(1/self.tf(word, sent))
        return i

    def vec(self, sent):
        word_list = re.split(r'\W*', sent.lower())
        vect = map(lambda word: self.tf(word, sent)*self.idf(word, sent), word_list)
        return vect
