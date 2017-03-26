import itertools
import re
import pandas as pd
import sys
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import json
from pymongo import MongoClient

def create_vocab(filename):
    data = pd.read_csv(filename)
    fields = ['question1', 'question2']
    data_q = data[fields]
    data_q_array = np.array(data_q).flatten()
    data_q_splitted = map(lambda x: re.split(r'\s\W*|\W*$', x) if isinstance(x,str) else x,data_q_array)
    vocab_all = []
    for i in data_q_splitted: 
        try:
	    vocab_all+=i 
	except Exception as e: print e,i
    vocab_all = list(set(vocab_all))
    return vocab_all

def create_trainable_for_vector(filenames):
    data_all = []
    for filename in filenames:
        data = pd.read_csv(filename)
        fields = ['question1', 'question2']
        data_q = data[fields]
	data_q_array = np.array(data_q).flatten()
	data_q_array = map(lambda x: re.sub(r'\W*$',' ',x) if isinstance(x, str) else '', data_q_array)
	data_all += data_q_array
    data = ' '.join(data_all)
    return data

def write_vectors(binary_file_path, vocab_file_path, client):
    word_vectors = KeyedVectors.load_word2vec_format(binary_file_path, binary=True)  # C binary format
    vocab = json.load(open(vocab_file_path))
    for i in vocab:
        try:
	    client.data.wordvec.insert({i: word_vectors[i].tolist()})
	except Exception as e:
	    print e


if __name__ == '__main__':
    client = MongoClient()
    write_vectors('../quora_data/quora-vector.bin', '../quora_data/vocab.json', client)
