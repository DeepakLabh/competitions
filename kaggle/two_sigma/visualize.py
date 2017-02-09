import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from scipy.stats.stats import pearsonr   

__saved_context__ = {}
#plt.rcParams['agg.path.chunksize'] = 1000
def saveContext():
    import sys
    __saved_context__.update(sys.modules[__name__].__dict__)

def restoreContext():
    import sys
    names = sys.modules[__name__].__dict__.keys()
    for n in names:
        if n not in __saved_context__:
            del sys.modules[__name__].__dict__[n]

def train_gen(data={}):
    for i in data.keys():
        yield [i,data[i]]

def plot_all(data_points):  
    fields = train.keys()
    #print [list(i) for i in train_gen(train)][:20]
    for i in train_gen(train):
        try:
            #saveContext()
            #train = pd.read_hdf('../kaggle_data/train.h5')
            # float(train[i][1])
            i = list(i)
            i[1] = map(lambda x: 0 if x==float('nan') else x,i[1])[:data_points]
            plt.plot(i[1])
            plt.savefig('plots/'+i[0]+'.png')
            #restoreContext()
            plt.clf()
            plt.close()
        except Exception as e:
            print e
            pass

def cor(length = 0):
    d = {}
    y = train.y
    for i in train_gen(train):
        j = list(i)
        # j[1] = map(lambda x: 0 if x==float('nan') else x,j[1])
        c = pearsonr(y[:length],j[1][:length])
        d[j[0]] = c
        # print j[0],c
    return d

def fields_relation(features, length):
    for i in features:
        data = train[i]

        p = plt.subplot(3,1,1)
        p.set_title(i, fontsize=1)
        p.plot(data[200:length])

        ########################  single diff ###############
        data_1 = data-data.shift(1)
        data_100 = data-data.shift(100)
        p = plt.subplot(3,1,2)
        p.set_title(i+ ':-diff 1', fontsize=1)
        p.plot(data_1[200:length])

        # p = plt.subplot(5,1,3)
        # p.set_title(i+ ':-diff 100')
        # p.plot(data_100[200:length])
        ########################  single diff ###############
        ########################  double diff ###############
        data__1 = data_1-data_1.shift(1)
        data__100 = data_100-data_100.shift(100)

        p = plt.subplot(3,1,3)
        p.set_title(i+ ':-double diff 1', fontsize=1)
        p.plot(data__1[200:length])

        # p = plt.subplot(5,1,5)
        # p.set_title(i+ ':-double diff 100')
        # p.plot(data__100[200:length])
        ########################  double diff ###############
        # plt.legend(fontsize=5)

        plt.savefig(i+'STATIONARIZE')
        plt.clf()


if __name__ == '__main__':
    train = pd.read_hdf('../kaggle_data/train.h5')
    mean_values = train.mean(axis=0)
    train.fillna(mean_values, inplace=True)
    # train.fillna(0)
    print ('na filled')
    if sys.argv[-1]=='plot':
        plot_all(1000)
    if sys.argv[-1] == 'cor':
        dict_cor = cor(1000)  
        # print dict_cor.keys() 
        dict_cor = sorted(dict_cor, key = lambda x: dict_cor[x][0], reverse = True)
        top_features = map(str, dict_cor[:10])
        print top_features
    if sys.argv[-1] == 'stats':
        features = train.keys()
        fields_relation(features, 300)