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
from NLPutilsDL import doc2vec
import utils
import tfidf
import threading

#import pickle as pk
#import gzip

def synchronized(func):

    func.__lock__ = threading.Lock()

    def synced_func(*args, **kws):
        with func.__lock__:
            return func(*args, **kws)

    return synced_func
@synchronized
def create_test_data(start_index, end_index, q1, q2, wordvec_dict, df):
    #print 'create test data'
    x1_train = []
    x2_train = []
    x_train = []
    #############################################
    ###################################3 For indexing question 1 ###################
    for i in range(start_index, end_index):
        q1[i] = ' '.join(q1[i])
        q2[i] = ' '.join(q2[i])
        try:
            q1_tf_vec, q2_tf_vec = map(lambda x: np.lib.pad(tf.vec(x)[:max_sent_len], (0,max(0,max_sent_len-len(tf.vec(x)))), 'constant', constant_values = (0)).reshape(max_sent_len,1), [q1[i], q2[i]])
            q1_vec = []
            q2_vec = []
            for j in xrange(max_sent_len):
                try:
                    if j>=len(q1[i]):
                        q1_vec.append(np.zeros(wordvec_dim))
                    else:
                        q1_vec.append(wordvec_dict[q1[i][j]])

                except Exception as e1:
                    q1_vec.append(np.zeros(wordvec_dim))

                try:
                    if j>=len(q2[i]):
                        q2_vec.append(np.zeros(wordvec_dim))
                    else:
                        q2_vec.append(wordvec_dict[q2[i][j]])
                except Exception as e1:
                    q2_vec.append(np.zeros(wordvec_dim))
            x1_train.append(np.array(q1_vec)*q1_tf_vec)
            x2_train.append(np.array(q2_vec)*q2_tf_vec)
        except:
            x1_train.append(x1_train[-1])
            x2_train.append(x2_train[-1])

        try: 1/(i%10000)
        except: print i
    return [np.array(x1_train), np.array(x2_train)]
    ###################################3 For indexing question 1 ###################

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
#############################
import json
test_data = pd.read_csv('../quora_data/train.csv')
tf = tfidf.tfidf('../quora_data/vec_train.txt')
print 'tfidf vector class initialized ...'
only_tfidf = False
if only_tfidf: wordvec_dim = 1
wordvec_dict = json.load(open('../quora_data/wordvec.dict'))
print 'wordvec dict loaded ...'
q1 = test_data['question1']
q2 = test_data['question2']
print 'questions data loaded ...'
q1 = map(lambda x: filter(lambda xx: len(xx)>0, re.split(r'\W*', str(x).lower())[:-1]) , q1)
print 'question 1 data prepared ...'
q2 = map(lambda x: filter(lambda xx: len(xx)>0, re.split(r'\W*', str(x).lower())[:-1]) , q2)
print 'question 2 data prepared ...'
print 'train data loaded, creating vectors'
#####################################

@synchronized
def batch_gen_2d(batch_size, start_index, end_index, only_tfidf = False, tfidf_x_wordvec = True):
    while True:
        i = randint(start_index, end_index)
        y_batch = np.array(map(lambda x: np.array([1,0] if x==1 else [0,1]),y[i:i+batch_size]))
        df = pd.DataFrame(columns = ['test_id', 'q1', 'q2', 'q1_tf', 'q2_tf'])
        df_out = create_test_data(i, i+batch_size, q1, q2, wordvec_dict, df)

        #dd = doc2vec.data('../quora_data/train.csv', 'csv', coll_vec)
        #x_batch2 = dd.create_vectors('question2', 'word', 'vec', max_sent_len = 50, max_num_sents = 4, wordvec_dim = wordvec_dim, sent_tokenize_flag = False, start_index=i, end_index=i+batch_size)
        #x_batch1 = dd.create_vectors('question1', 'word', 'vec', max_sent_len = 50, max_num_sents = 4, wordvec_dim = wordvec_dim, sent_tokenize_flag = False, start_index=i, end_index=i+batch_size)
        ############# To normalize the inputs ############3

        #if only_tfidf or tfidf_x_wordvec:
        #    x_batch1_tf = map(lambda x: np.array(x[:max_sent_len] if len(x)>= max_sent_len else x+list(np.zeros(max_sent_len-len(x)))), dd.create_tfidf_vectors('question1', i , i+batch_size))
        #    x_batch1_tf = map(lambda x: np.array(x[:max_sent_len] if len(x)>= max_sent_len else x+list(np.zeros(max_sent_len-len(x)))), dd.create_tfidf_vectors('question2', i , i+batch_size))
        #    x_batch1_tf = np.array(x_batch1_tf).reshape(batch_size, max_sent_len, 1)
        #    x_batch2_tf = np.array(x_batch2_tf).reshape(batch_size, max_sent_len, 1)
        #    if only_tfidf:
        #        yield [x_batch1_tf, x_batch2_tf], y_batch
        #        continue

        ##x_batch1 = map(lambda x: map(lambda xx: abs(xx)-abs(x.mean()), x), x_batch1)
        ##x_batch2 = map(lambda x: map(lambda xx: abs(xx)-abs(x.mean()), x), x_batch2)
        ############# To normalize the inputs ############3
        #x_batch1, x_batch2 = [np.array(x_batch1), np.array(x_batch2)]
        #############################  for vec * tfidf ################################
        #if tfidf_x_wordvec:
        #    yield ([x_batch1*x_batch1_tf, x_batch2*x_batch2_tf], y_batch)
        #    continue
        #############################  for vec * tfidf ################################
        #yield ([x_batch1, x_batch2], y_batch)
        yield (df_out, y_batch)
        # break


model = ma.siamese(max_sent_len, wordvec_dim, gru_output_dim, output_dim)
model.load_weights('../quora_data/best.weights')
print 'fitting model ...'
only_tfidf = False
if sys.argv[-1]=='only_tfidf':
    model = ma.siamese(max_sent_len, 1 , gru_output_dim, output_dim)
    only_tfidf = True
checkpoint = ModelCheckpoint('../quora_data/best.weights', monitor='val_acc', save_best_only=True, verbose=2)
#fit = model.fit_generator(generator=batch_gen_2d(500,1,10000), nb_epoch=30, samples_per_epoch=train_size)
history = model.fit_generator(generator=batch_gen_2d(batch_size,1,390000,only_tfidf),nb_epoch = num_epoch, validation_data = batch_gen_2d(batch_size,395000,398010, only_tfidf), nb_val_samples = 1000, samples_per_epoch=train_size, callbacks=[checkpoint])

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



