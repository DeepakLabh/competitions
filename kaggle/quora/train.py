import pandas as pd
#from keras.utils import plot_model
import matplotlib
from pymongo import MongoClient
import numpy as np
import model_arch as ma
import sys
import re
from random import randint
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

#import pickle as pk
#import gzip

matplotlib.use('Agg')
c = MongoClient()
coll_vec = c.data.wordvec
question1_collection = c.data.question1
question2_collection = c.data.question2
data = pd.read_csv('../quora_data/train.csv')
y = data['is_duplicate']
max_sent_len = 30
wordvec_dim = 200
gru_output_dim = 50
output_dim = 2
train_size = 5000
batch_size = 1000
num_epoch = 100
def batch_gen_2d(batch_size, start_index, end_index, only_tfidf = False, tfidf_x_wordvec = True):
    while True:
        i = randint(start_index, end_index)
        #i = randint(1, 1000)
	#i = 1
	#print '    ',i,' iiiiiiiiiii2'
        y_batch = np.array(map(lambda x: np.array([1,0] if x==1 else [0,1]),y[i:i+batch_size]))
        x_batch2 = map(lambda x: np.array(x['vec'][:max_sent_len]), list(question2_collection.find({'_id': {'$gte': i, '$lt': i+batch_size}})))
        x_batch1 = map(lambda x: np.array(x['vec'][:max_sent_len]), list(question1_collection.find({'_id': {'$gte': i, '$lt': i+batch_size}})))
        #x_batch = map(lambda x: np.reshape(np.array(x), (input_shape[0], x_train.shape[1])),x_batch) # 10 is nu of recurrent layers
        ############ To normalize the inputs ############3

        if only_tfidf or tfidf_x_wordvec:
            x_batch1_tf = map(lambda x: np.array(x['tf'][:max_sent_len] if len(x['tf'])>= max_sent_len else x['tf']+list(np.zeros(max_sent_len-len(x['tf'])))), list(question1_collection.find({'_id': {'$gte': i, '$lt': i+batch_size}})))
            x_batch2_tf = map(lambda x: np.array(x['tf'][:max_sent_len] if len(x['tf'])>= max_sent_len else x['tf']+list(np.zeros(max_sent_len-len(x['tf'])))), list(question2_collection.find({'_id': {'$gte': i, '$lt': i+batch_size}})))
            x_batch1_tf = np.array(x_batch1_tf).reshape(batch_size, max_sent_len, 1)
            x_batch2_tf = np.array(x_batch2_tf).reshape(batch_size, max_sent_len, 1)
            if only_tfidf:
                yield [x_batch1_tf, x_batch2_tf], y_batch
                continue

        x_batch1 = map(lambda x: map(lambda xx: abs(xx)-abs(x.mean()), x), x_batch1)
        x_batch2 = map(lambda x: map(lambda xx: abs(xx)-abs(x.mean()), x), x_batch2)
        ############ To normalize the inputs ############3
        x_batch1, x_batch2 = [np.array(x_batch1), np.array(x_batch2)]
        #print x_batch1.shape, x_batch1_tf.shape, x_batch2.shape, x_batch2_tf.shape
        #x_batch1 = np.array(map(lambda x,yy:np.multiply(x.transpose(),yy).transpose(), x_batch1, x_batch1_tf))
        #x_batch2 = np.array(map(lambda x,yy:np.multiply(x.transpose(),yy).transpose(), x_batch2, x_batch2_tf))
        ############################  for vec * tfidf ################################
        #print y_batch,111111111111111
        if tfidf_x_wordvec:
            yield ([x_batch1*x_batch1_tf, x_batch2*x_batch2_tf], y_batch)
            continue
        ############################  for vec * tfidf ################################
        yield ([x_batch1, x_batch2], y_batch)
        # break


model = ma.siamese(max_sent_len, wordvec_dim, gru_output_dim, output_dim)

print 'fitting model ...'
only_tfidf = False
if sys.argv[-1]=='only_tfidf':
    model = ma.siamese(max_sent_len, 1 , gru_output_dim, output_dim)
    only_tfidf = True
checkpoint = ModelCheckpoint('../quora_data/best.weights', monitor='val_acc', save_best_only=True, verbose=2)
#fit = model.fit_generator(generator=batch_gen_2d(500,1,10000), nb_epoch=30, samples_per_epoch=train_size)
history = model.fit_generator(generator=batch_gen_2d(batch_size,1,90000,only_tfidf),nb_epoch = num_epoch, validation_data = batch_gen_2d(batch_size,95000,98010, only_tfidf), nb_val_samples = 1000, samples_per_epoch=train_size, callbacks=[checkpoint])

#plot_model(model, to_file='model.png')

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



