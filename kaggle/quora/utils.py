import itertools
import re
import pandas as pd
import sys
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import json
from pymongo import MongoClient
from multiprocessing import Process
import tfidf

def create_vocab(filename):
    data = pd.read_csv(filename)
    fields = ['question1', 'question2']
    data_q = data[fields]
    data_q_array = np.array(data_q).flatten()
    data_q_splitted = map(lambda x: re.split(r'\W*', x) if isinstance(x,str) else '',data_q_array)
    vocab_all = []
    for i in data_q_splitted:
        try:
	    vocab_all+=i
	except Exception as e: print e,i
    vocab_all = map(lambda x: x.lower(), list(set(vocab_all)))

    return vocab_all

def create_trainable_for_vector(filenames):
    data_all = []
    for filename in filenames:
        data = pd.read_csv(filename)
        fields = ['question1', 'question2']
        data_q = data[fields]
	data_q_array = np.array(data_q).flatten()
	data_q_array = map(lambda x: re.sub(r'\W+',' ',x)+'. ' if isinstance(x, str) else '', data_q_array)
	data_all += data_q_array
    data = ' '.join(data_all).lower()
    return data

def generate_vectors(word_vectors, word, text_file_path, c):
    try:
	vec = word_vectors[word]
    except Exception as ee:
	print ee,11111111

        count = 0
        a = open(text_file_path).read()
        vec = np.zeros(wordvec_dim)
        try:
            nearby_words = filter(lambda x: word in x, re.split(r'\s{2,}',a[a.index(word)-100:a.index(word)+100]))[0].split()
            for i in nearby_words:
                try:
            	    vec += word_vectors[i]
            	    count+=1
                except:
            	    pass
            vec = vec/count
        except Exception as e:
            print 'cannot synthesize new vectors ',e
            vec = np.zeros(wordvec_dim)

	c.data.wordvec.insert({'vec': vec.tolist(), 'word':word})
        #return vec



def write_vectors(binary_file_path, vocab_file_path, c):
    word_vectors = KeyedVectors.load_word2vec_format(binary_file_path, binary=True)  # C binary format
    vocab = json.load(open(vocab_file_path))
    mm = map(lambda x: generate_vectors(word_vectors, x.lower(), '../quora_data/vec_train.txt', c), vocab)
#    for i in vocab:
#        try:
#	    c.data.wordvec.insert({'vec': word_vectors[i].tolist(), 'word':i})
#	except Exception as e:
#	    c.data.wordvec.insert({'vec': generate_vectors(word_vectors, i, '../quora_data/vec_train.txt').tolist(), 'word':i})
#	    print e

def predict_similarity(question1, question2, vec_coll, model, only_tfidf=False, batch_size = 1):
    #print question1, question2
    #wordvec_dim = 200
    q = map(lambda x: re.sub(r'\W+',' ',x.lower()) if isinstance(x, str) else '', [question1, question2])
    if only_tfidf:
        q_vec = map(lambda x: np.lib.pad(tf.vec(x)[:max_sent_len], (0,max(0,max_sent_len-len(tf.vec(x)))), 'constant', constant_values = (0)).reshape(1,max_sent_len,1), q)
        return model.predict(q_vec)
    if batch_size > 1:
        pass
    q_vec = map(lambda x: map(lambda xx:np.array(vec_coll.find_one({'word': xx})['vec']) if vec_coll.find_one({'word': xx}) else np.zeros(wordvec_dim), x), q)
    q_vec = map(lambda x: x[:max_sent_len] if len(x) > max_sent_len else np.concatenate((x,np.zeros((max_sent_len-len(x), wordvec_dim))), axis = 0), q_vec)
    #q_vec = map(lambda x: np.array(x).reshape((1, max_sent_len, wordvec_dim)), q_vec)
    ##############################  load TFIDF vectors ##########
    q_vec_tf = map(lambda x: np.lib.pad(tf.vec(x)[:max_sent_len], (0,max(0,max_sent_len-len(tf.vec(x)))), 'constant', constant_values = (0)).reshape(max_sent_len,1), q)
    ##############################  load TFIDF vectors ##########
    #print np.array(q_vec).shape, np.array(q_vec_tf).shape, 111111111, q_vec_tf
    q_vec = map(lambda x,y: x*y, q_vec, q_vec_tf)
    q_vec = map(lambda x: np.array(x).reshape((1, max_sent_len, wordvec_dim)), q_vec)
    #print q_vec
    out = model.predict(q_vec)
    return out

def index_training_data(num_data):
    print 'create trainable data'
    #x1 = map(lambda x: map(lambda xx: coll_vec.find_one({'word':xx})['vec'] if coll_vec.find_one({'word':xx})!=None else np.zeros(wordvec_dim), x), x1)
    #x2 = map(lambda x: map(lambda xx: coll_vec.find_one({'word':xx})['vec'] if coll_vec.find_one({'word':xx})!=None else np.zeros(wordvec_dim), x), x2)
    x1_train = []
    #######################################3 For indexing question 1 ###################
    for i in xrange(num_data):
        x1_tfidf = tf.vec(' '.join(x1[i]))
        #print 'hello111111111',i
        x1_sents = []
        for j in xrange(max_sent_len):
            try:
                if j>=len(x1[i]):
                    x1_sents.append(np.zeros(wordvec_dim))
                else:
                    x1_sents.append(coll_vec.find_one({'word':x1[i][j]})['vec'])

            except:
               #if j>=len(x1[i]):
                x1_sents.append(np.zeros(wordvec_dim))
               # else:
               #     x1[i][j] = np.zeros(wordvec_dim)
        question1_collection.insert_one({'_id':i, 'vec':np.array(x1_sents).tolist(), 'tf':x1_tfidf})
        try: 1/(i%10000)
        except: print i
    #######################################3 For indexing question 1 ###################

    #######################################3 For indexing question 2 ###################
    for i in xrange(num_data):
        x2_tfidf = tf.vec(' '.join(x2[i]))
        x2_sents = []
        for j in xrange(max_sent_len):
            try:
                if j>=len(x2[i]):
                    x2_sents.append(np.zeros(wordvec_dim))
                else:
                    x2_sents.append(coll_vec.find_one({'word':x2[i][j]})['vec'])
            except:
                #if j>=len(x2[i]):
                x2_sents.append(np.zeros(wordvec_dim))
                #else:
                #    x2[i][j] = np.zeros(wordvec_dim)
        question2_collection.insert_one({'_id':i, 'vec':np.array(x2_sents).tolist(), 'tf':x2_tfidf})
        try: 1/(i%10000)
        except: print i
    #######################################3 For indexing question 2 ###################

def create_test_data(start_index, end_index, q1, q2, wordvec_dict, df):
    print 'create test data'
    #x1 = map(lambda x: map(lambda xx: coll_vec.find_one({'word':xx})['vec'] if coll_vec.find_one({'word':xx})!=None else np.zeros(wordvec_dim), x), x1)
    #x2 = map(lambda x: map(lambda xx: coll_vec.find_one({'word':xx})['vec'] if coll_vec.find_one({'word':xx})!=None else np.zeros(wordvec_dim), x), x2)
    x1_train = []
    x2_train = []
    #############################################

    #df = df.astype(int)
    #############################################
    #######################################3 For indexing question 1 ###################
    for i in range(start_index, end_index):
        #q1_tf_vec = tf.vec(' '.join(q1[i]))
        q1_tf_vec = map(lambda x: np.lib.pad(tf.vec(x)[:max_sent_len], (0,max(0,max_sent_len-len(tf.vec(x)))), 'constant', constant_values = (0)).reshape(max_sent_len,1), q1[i])
        q1_vec = []
        #q2_tf_vec = tf.vec(' '.join(q2[i]))
        q2_tf_vec = map(lambda x: np.lib.pad(tf.vec(x)[:max_sent_len], (0,max(0,max_sent_len-len(tf.vec(x)))), 'constant', constant_values = (0)).reshape(max_sent_len,1), q2[i])
        q2_vec = []
        for j in xrange(max_sent_len):
            try:
                if j>=len(q1[i]):
                    q1_vec.append(np.zeros(wordvec_dim))
                else:
                    q1_vec.append(wordvec_dict[x1[i][j]])

            except:
               #if j>=len(x1[i]):
                q1_vec.append(np.zeros(wordvec_dim))
               # else:
               #     x1[i][j] = np.zeros(wordvec_dim)

            try:
                if j>=len(q2[i]):
                    q2_vec.append(np.zeros(wordvec_dim))
                else:
                    q2_vec.append(wordvec_dict[x2[i][j]])
            except:
                #if j>=len(x2[i]):
                q2_vec.append(np.zeros(wordvec_dim))
                #else:
                #    x2[i][j] = np.zeros(wordvec_dim)
        print np.array(q1_tf_vec).shape, np.array(q1_vec).shape
        x1_train.append(np.array(q1_vec)*q1_tf_vec)
        x2_train.append(np.array(q2_vec)*q2_tf_vec)
        #df1 = pd.DataFrame([[int(test_data['test_id'][i]), q1_vec, q2_vec, q1_tf_vec, q2_tf_vec]], columns = ['test_id', 'q1', 'q2', 'q1_tf', 'q2_tf'])
        #df = df.append(df1)
        #question1_collection.insert_one({'_id':i, 'vec':np.array(x1_sents).tolist(), 'tf':x1_tfidf})
        try: 1/(i%100)
        except: print i
    return np.array([x1_train, x2_train])
    #######################################3 For indexing question 1 ###################


if __name__ == '__main__':
    c = MongoClient()
    coll_vec = c.data.wordvec
    question1_collection = c.data.question1
    question2_collection = c.data.question2
    wordvec_dim = 200
    max_sent_len = 30
    gru_output_dim = 50
    output_dim = 2
    if sys.argv[-1] == 'create_vocab': ### Create vocab of both train and test data
        d1= create_vocab('../quora_data/train.csv')
        d2= create_vocab('../quora_data/test.csv')
        d = list(set(d1+d2))
        json.dump(d, open('../quora_data/vocab.json','w'))

    if sys.argv[-1] == 'index_vectors': ### Write word vectors into mongodb
        #write_vectors('../quora_data/quora-vector.bin', '../quora_data/vocab.json', c)
        p = Process(target=write_vectors, args=('../quora_data/quora-vector.bin', '../quora_data/vocab.json', c))
        p.start()
        p.join()
        #p.join()

    if sys.argv[-1]=='index_train':### Create training data and index in mongodb
        import pandas as pd
        from pymongo import MongoClient
        import numpy as np
        import re
        tf = tfidf.tfidf('../quora_data/vec_train.txt')
        print 'tfidf vectorizer loaded'
        data = pd.read_csv('../quora_data/train.csv')
        y = data['is_duplicate']
        x1 = data['question1']
        x2 = data['question2']

        x1 = map(lambda x: filter(lambda xx: len(xx)>0, re.split(r'\W*', str(x).lower())[:-1]) , x1)
        x2 = map(lambda x: filter(lambda xx: len(xx)>0, re.split(r'\W*', str(x).lower())[:-1]) , x2)


        max_sent_len = max([max(map(lambda x: len(x), x1)), max(map(lambda x: len(x), x2))])
        max_sent_len = 60

        index_training_data(100000)

    if sys.argv[-1] == 'predict': ### Sample prediction
	import model_arch as ma
        tf = tfidf.tfidf('../quora_data/vec_train.txt')
	q1,q2 = sys.argv[-3], sys.argv[-2]
        only_tfidf = False
        if only_tfidf: wordvec_dim = 1
	model = ma.siamese(max_sent_len, wordvec_dim, gru_output_dim, output_dim)
	model.load_weights('../quora_data/best.weights')
	out = predict_similarity(q1, q2, coll_vec, model, only_tfidf)
	print out,'  <<-------- OUTPUT'

    if sys.argv[-1] == 'create_vec_data': ### Create trainables for vector
	data = create_trainable_for_vector(['../quora_data/train.csv', '../quora_data/test.csv'])
	open('../quora_data/vec_train.txt', 'w').write(data)

    if sys.argv[-1] == 'sub': # for submission
        test_data = pd.read_csv('../quora_data/test.csv')
	import model_arch as ma
        only_tfidf = False
        tf = tfidf.tfidf('../quora_data/vec_train.txt')
	#q1,q2 = sys.argv[-3], sys.argv[-2]
        if only_tfidf: wordvec_dim = 1
	model = ma.siamese(max_sent_len, wordvec_dim, gru_output_dim, output_dim)
	model.load_weights('../quora_data/best.weights')

        df = pd.DataFrame(columns = ['test_id', 'is_duplicate'])
        #for i in xrange(len(test_data['id'])):
        i = 0
        while i<len(test_data):
            #batch_data = test_data[i:i+batch_size]
            q1 = test_data['question1'][i]
            q2 = test_data['question2'][i]
	    out = predict_similarity(q1, q2, coll_vec, model, only_tfidf)
            out = list(out[0])
            out = out.index(min(out))
            df1 = pd.DataFrame([[int(test_data['test_id'][i]), int(out)]], columns = ['test_id', 'is_duplicate'])
            df = df.append(df1)
            i=i+1
            if i%10==0 : print i, df1, q1, q2,111111111
            #if i%10000==0: break

        df = df.astype(int)
        df.to_csv('../quora_data/Sample_submission_1.csv', index=False)


    if sys.argv[-1] == 'create_test_data': # for submission
        import json
        test_data = pd.read_csv('../quora_data/test.csv')
        tf = tfidf.tfidf('../quora_data/vec_train.txt')
	import model_arch as ma
        only_tfidf = False
        if only_tfidf: wordvec_dim = 1
	model = ma.siamese(max_sent_len, wordvec_dim, gru_output_dim, output_dim)
	model.load_weights('../quora_data/best.weights')
	#q1,q2 = sys.argv[-3], sys.argv[-2]
        if only_tfidf: wordvec_dim = 1
        df = pd.DataFrame(columns = ['test_id', 'q1', 'q2', 'q1_tf', 'q2_tf'])
        wordvec_dict = json.load(open('../quora_data/wordvec.dict'))
        q1 = test_data['question1']
        q2 = test_data['question2']
        q1 = map(lambda x: filter(lambda xx: len(xx)>0, re.split(r'\W*', str(x).lower())[:-1]) , q1)
        q2 = map(lambda x: filter(lambda xx: len(xx)>0, re.split(r'\W*', str(x).lower())[:-1]) , q2)
        print 'test data loaded, creating vectors'
        df_out = create_test_data(0,1000, q1, q2, wordvec_dict, df)
        out = model.predict(df_out, batch_size=1000)
        print out
