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
batch_size = 40

def batch_gen_2d(batch_size, start_index, end_index):
    while True:
        i = randint(start_index, end_index)
        y_batch = np.array(map(lambda x: np.array([1,0] if x==1 else [0,1]),y[i:i+batch_size]))
        x_batch2 = map(lambda x: np.array(x['vec']), list(question2_collection.find({'_id': {'$gte': i, '$lt': i+batch_size}})))	
        x_batch1 = map(lambda x: np.array(x['vec']), list(question1_collection.find({'_id': {'$gte': i, '$lt': i+batch_size}})))	
        #x_batch = map(lambda x: np.reshape(np.array(x), (input_shape[0], x_train.shape[1])),x_batch) # 10 is nu of recurrent layers
        
        x_batch1 = np.array(list(x_batch1))
        x_batch2 = np.array(list(x_batch2))
        
        yield [x_batch1, x_batch2], y_batch
        # break


model = ma.siamese(max_sent_len, wordvec_dim, gru_output_dim, output_dim)
print 'fitting model ...'

#fit = model.fit_generator(generator=batch_gen_2d(500,1,10000), nb_epoch=30, samples_per_epoch=train_size)
history = model.fit_generator(generator=batch_gen_2d(batch_size,1,44000), nb_epoch=200, validation_data = batch_gen_2d(batch_size,46000,48000), nb_val_samples = 500, samples_per_epoch=train_size)

model.save_weights("../quora_data/siamese_lstm.weights",overwrite=True) 

#########################3 PLOT loss and accuracy ##############
# summarize history
pars = history.history.keys()
for par in pars:
    if 'val_' in par:
        act_par = re.findall('val_(.*)', par)[0]
        plt.plot(history.history[act_par])
        plt.plot(history.history[par])
        plt.title('model '+act_par)
        plt.ylabel(act_par)
        plt.xlabel('epoch')
        plt.legend(['train', 'validate'], loc='upper right')
        # plt.show()
        plt.savefig(str(act_par+'.png'))
        plt.cla()
        #plt.close()
#########################3 PLOT loss and accuracy ##############
print "Training Completed"
