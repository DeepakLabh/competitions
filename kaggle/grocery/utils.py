from datetime import datetime
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras

#class Plotter(object, self):
#    def __init__(self, data):
#        slef.data = data
#
#    def plot(self):
#        plt.plot(d)

def vis1():
    chunksize = 10
    sub2 = pd.read_csv("/home/arya/deepak/competitions/kaggle/grocery/submission_data/test_ma8dwof.csv")
    #for sub1,sub2 in [pd.read_csv("/home/arya/deepak/competitions/kaggle/grocery/submission_data/keras_submission.csv", chunksize=chunksize), pd.read_csv("/home/arya/deepak/competitions/kaggle/grocery/submission_data/test_ma8dwof.csv", chunksize=chunksize)]:
    count = 0
    for sub1 in pd.read_csv("/home/arya/deepak/competitions/kaggle/grocery/submission_data/keras_submission.csv", chunksize=chunksize):
        plt.plot(sub1['unit_sales'], label = 'prd')
        plt.plot(range(0,chunksize), sub2['unit_sales'][count:count+chunksize], label='benchmark')
        count+=chunksize
        plt.ion()
        plt.show()
        plt.pause(0.001)
        time.sleep(1)
        #num = input('press any key to move to other frame: ')
        plt.clf()
        plt.close('all')

def vis2(actual, predicted):
    plt.plot(predicted, label = 'prd')
    plt.plot(range(0,len(actual)), actual, label = 'actual')
    plt.ion()
    plt.show()
    plt.pause(0.001)
    time.sleep(0.01)
    plt.clf()
    #plt.close('all')


def prepare_data(data):
    
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    
    data['sin_month'] = np.sin(2*np.pi*data['date'].dt.month/12)
    data['sin_day'] = np.sin(2*np.pi*data['date'].dt.day/31)
    data['sin_dayofweek'] = np.sin(2*np.pi*data['date'].dt.dayofweek/7)
    
    data['cos_month'] = np.cos(2*np.pi*data['date'].dt.month/12)
    data['cos_day'] = np.cos(2*np.pi*data['date'].dt.day/31)
    data['cos_dayofweek'] = np.cos(2*np.pi*data['date'].dt.dayofweek/7)
    
    data['unit_sales'] =  data['unit_sales'].apply(pd.np.log1p) #logarithm conversion
    
    item_nbr_bin = data['item_nbr'].apply(lambda x: [i for i in bin(x)[2:]])
    item_nbr_bin = keras.preprocessing.sequence.pad_sequences(item_nbr_bin, maxlen=25, dtype='int32', padding='pre', truncating='pre', value=0.)
    
    store_nbr_bin = data['store_nbr'].apply(lambda x: [i for i in bin(x)[2:]])
    store_nbr_bin = keras.preprocessing.sequence.pad_sequences(store_nbr_bin, maxlen=10, dtype='int32', padding='pre', truncating='pre', value=0.)
    x_data = [np.array(data.drop(['id','date','unit_sales','onpromotion', 'item_nbr','store_nbr', 'year'], axis=1)), item_nbr_bin, store_nbr_bin]
    y_data = np.array(data['unit_sales'])
    return (x_data, y_data)


def time_series(data, field_name, field_value):
    pass 


def write_to_mongo(data, coll):
    dict_keys = data.keys()
    for i in xrange(len(data)):
        edict = {}
        for keys in dict_keys:
            edict.update({keys: data.iloc[i][keys]})
        date = edict['date'].split('-')
        edict['date'] = datetime(int(date[0]), int(date[1]), int(date[2]))
        coll.insert(edict)


if __name__ == '__main__':
    if sys.argv[-1]=='write_to_mongo':
        from pymongo import MongoClient
        coll = MongoClient().grocery.train
        
        CHUNKSIZE = 120000
        TRAIN_PATH = 'data/train.csv'
        count = 0
        for train in pd.read_csv(TRAIN_PATH, chunksize=CHUNKSIZE):
            count+=1
            if count<100:
                print (count)
                continue
            write_to_mongo(train, coll)
