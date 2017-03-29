import pandas as pd
from pymongo import MongoClient
import numpy as np
import model_arch as ma
import re

c = MongoClient()
coll_vec = c.data.wordvec
wordvec_dim = 200

print 'load training data'
data = pd.read_csv('../quora_data/train.csv')
y = data['is_duplicate']
x1 = data['question1']
x2 = data['question2']

x1 = map(lambda x: re.split(r'\W*', x.lower())[:-1] if isinstance(x, str) else '', x1)
x2 = map(lambda x: re.split(r'\W*', x.lower())[:-1] if isinstance(x, str) else '', x2)


max_sent_len = max([max(map(lambda x: len(x), x1)), max(map(lambda x: len(x), x2))])
max_sent_len = 60

print 'create trainable data'
#x1 = map(lambda x: map(lambda xx: coll_vec.find_one({'word':xx})['vec'] if coll_vec.find_one({'word':xx})!=None else np.zeros(wordvec_dim), x), x1)
#x2 = map(lambda x: map(lambda xx: coll_vec.find_one({'word':xx})['vec'] if coll_vec.find_one({'word':xx})!=None else np.zeros(wordvec_dim), x), x2)

x1_train = []
for i in xrange(len(x1)):
    x1_sents = []
    for j in xrange(max_sent_len):
        try:
            if j>=len(x1[i]):
                x1_sents.append(np.zeros(wordvec_dim))
            else:
                x1_sents.append(coll_vec.find_one({'word':x1[i][j]})['vec'])
        except:
            if j>=len(x1[i]):
                x1_sents.append(np.zeros(wordvec_dim))
            else:
                x1_sents.append(np.zeros(wordvec_dim))
    try: i-1000/100
    except: print i

    #for j in xrange(max_sent_len):
    #    try:
    #        if j>=len(x2[i]):
    #            x2[i].append(np.zeros(wordvec_dim))
    #        else:
    #            x2[i][j] = coll_vec.find_one({'word':x2[i][j]})['vec']
    #    except:
    #        if j>=len(x2[i]):
    #    	x2[i].append(np.zeros(wordvec_dim))
    #        else:
    #            x2[i][j] = np.zeros(wordvec_dim)
    #try: i-1000/100
    #except: print i
