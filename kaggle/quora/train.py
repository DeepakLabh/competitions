import pandas as pd
from pymongo import MongoClient
import numpy as np
import model_arch as ma
import re
from random import randint

#import pickle as pk
#import gzip

c = MongoClient()
coll_vec = c.data.wordvec
question1_collection = c.data.question1
question2_collection = c.data.question2
data = pd.read_csv('../quora_data/train.csv')
y = data['is_duplicate']
max_sent_len = 60
wordvec_dim = 200
gru_output_dim = 100
output_dim = 2
train_size = 2000

def batch_gen_2d(batch_size, start_index, end_index):
    while True:
        i = randint(start_index, end_index)
        y_batch = np.array(map(lambda x: np.array([1,0] if x==1 else [0,1]),y[i:i+batch_size]))
	#y_batch = y_batch.reshape(batch_size, 1)
        x_batch2 = map(lambda x: np.array(x['vec']), list(question2_collection.find({'_id': {'$gte': i, '$lt': i+batch_size}})))	
        x_batch1 = map(lambda x: np.array(x['vec']), list(question1_collection.find({'_id': {'$gte': i, '$lt': i+batch_size}})))	
        #x_batch = map(lambda x: np.reshape(np.array(x), (input_shape[0], x_train.shape[1])),x_batch) # 10 is nu of recurrent layers
        # print (x_batch.shape,4444444444444444)
        
        x_batch1 = np.array(list(x_batch1))
        x_batch2 = np.array(list(x_batch2))
        
	#print 'tttttttttttttt', x_batch1.shape, x_batch2.shape, y_batch.shape
        yield [x_batch1, x_batch2], y_batch
        # break


model = ma.siamese(max_sent_len, wordvec_dim, gru_output_dim, output_dim)
print 'fitting model ...'

#fit = model.fit_generator(generator=batch_gen_2d(500,1,10000), nb_epoch=30, samples_per_epoch=train_size)
fit = model.fit_generator(generator=batch_gen_2d(500,1,10000), nb_epoch=30, validation_data = batch_gen_2d(500,10000,20000), nb_val_samples = 500, samples_per_epoch=train_size)


