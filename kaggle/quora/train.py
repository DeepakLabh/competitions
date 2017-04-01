import pandas as pd
from pymongo import MongoClient
import numpy as np
import model_arch as ma
import re

#import pickle as pk
#import gzip

c = MongoClient()
coll_vec = c.data.wordvec
question1_collection = c.data.question1
question2_collection = c.data.question2

def batch_gen_2d(batch_size):
    while True:
        i = randint(start_index, end_index)
        y_batch = np.array(y_train[i:i+batch_size])
        x_batch = ([x_train[k-input_shape[0]+1:k+1] for k in range(i, i+batch_size)]) #  Create 2-D input
        # x_batch = np.array(x_batch)
        # print (len(x_batch),33333333333, len(x_batch[0]))
        # x_batch = ([[x_train[k-j:k+1] for j in range(10,0,-1)] for k in range(i, i+batch_size-1)])
        # print (len(x_batch), len(x_batch[0]), len(x_batch[0].keys())),3333333333333333
        x_batch = map(lambda x: np.reshape(np.array(x), (input_shape[0], x_train.shape[1])),x_batch) # 10 is nu of recurrent layers
        # print (x_batch.shape,4444444444444444)
        
        x_batch = np.array(list(x_batch))
        

        yield (x_batch, y_batch)
        # break

