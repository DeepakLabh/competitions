'''
## TODO -
Cyclic transform of dates - done
Encoding or embeeding of Store num and product num = possibly in a same network
'''
import sys
import pandas as pd
import numpy as np
import random
import xgboost as xgb
# import utils as vs

num_items = 3127115
num_stores = 54
embedding_size = 30
#############################################################################################################################################
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LSTM, Lambda, Merge, Reshape, GRU, Input, merge, Flatten, Embedding
import keras
#from keras.optimizers import SGD
from keras.optimizers import SGD, RMSprop
#from keras.regularizers import l2, activity_l2, l1
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import optimizers
from keras.models import load_model
from keras.models import save_model
# import prepare_data as p

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# tra = pd.read_csv('/home/arya/deepak/competitions/kaggle/grocery/data/train_100000.csv')
hol = pd.read_csv('/home/arya/deepak/competitions/kaggle/grocery/data/holidays_events.csv')
itm = pd.read_csv('/home/arya/deepak/competitions/kaggle/grocery/data/items.csv')
st = pd.read_csv('/home/arya/deepak/competitions/kaggle/grocery/data/stores.csv')
oil = pd.read_csv('/home/arya/deepak/competitions/kaggle/grocery/data/oil.csv')
trans = pd.read_csv('/home/arya/deepak/competitions/kaggle/grocery/data/transactions.csv')

# hol = pd.read_csv('../input/holidays_events.csv')
# itm = pd.read_csv('../input/items.csv')
# st = pd.read_csv('../input/stores.csv')
# oil = pd.read_csv('../input/oil.csv')
# trans = pd.read_csv('../input/transactions.csv')

set_hol = array(list(set(map(lambda x,y:str(x)+str(y), hol['type'], hol['locale']))))
set_itm = array(list(set(map(lambda x:str(x), itm['family']))))
set_st =  array(list(set(map(lambda x,y:str(x)+str(y), st['city'], st['type']))))


hol_int_encoder, itm_int_encoder, st_int_encoder = [LabelEncoder(), LabelEncoder(), LabelEncoder()]
hol_int, itm_int, st_int = [hol_int_encoder.fit_transform(set_hol), itm_int_encoder.fit_transform(set_itm), st_int_encoder.fit_transform(set_st)]
hol_int, itm_int, st_int = [hol_int.reshape(len(hol_int),1), itm_int.reshape(len(itm_int),1), st_int.reshape(len(st_int),1)]

hol_encoder, itm_encoder, st_encoder = [OneHotEncoder(sparse=False), OneHotEncoder(sparse=False), OneHotEncoder(sparse=False)]
hol_encoder = dict(zip(set_hol, hol_encoder.fit_transform(hol_int)))
itm_encoder = dict(zip(set_itm, itm_encoder.fit_transform(itm_int)))
st_encoder =  dict(zip(set_st, st_encoder.fit_transform(st_int)))

def prepare_cat_data(data):
    mer = pd.merge(hol,data, how='right', on=['date'])
    mer = pd.merge(itm,mer, how='right', on=['item_nbr'])
    mer = pd.merge(st,mer, how='right', on=['store_nbr'])
    mer = pd.merge(oil,mer, how='right', on=['date'])
    mer = pd.merge(trans,mer, how='right', on=['date', 'store_nbr'])

    hol_encoded = (mer['type_y'] + mer['locale']).apply(lambda x: list(hol_encoder[str(x)]) if 'nan' not in str(x).lower() else list(np.zeros(len(list(hol_encoder.values())[0]))))
    itm_encoded = (mer['family']).apply(lambda x: list(itm_encoder[str(x)]) if 'nan' not in str(x).lower() else list(np.zeros(len(list(itm_encoder.values())[0]))))
    st_encoded = (mer['city'] + mer['type_x']).apply(lambda x: list(st_encoder[str(x)]) if 'nan' not in str(x).lower() else list(np.zeros(len(list(st_encoder.values())[0]))))
    cat = hol_encoded + itm_encoded + st_encoded + mer['perishable'].apply(lambda x:[float(x)])
    cat = list(map(lambda x: np.array(x), cat))
    #mer['cat'] = mer['cat'].apply(lambda x: array(x))
    #print np.asarray(cat['cat']),111111111111111111111111
    return np.array(cat)
    


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

def acc_loss(y_true, y_pred):
    rms =  K.sqrt(K.mean(K.square(y_true - y_pred)))
    return rms
def nrms(y_true, y_pred):
    rms = np.sqrt(np.mean(np.square(y_true - y_pred)))
    return rms
    
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def fcnn(input_dim1, input_dim2, input_dim3, input_dim4):
    input_layer1 = Input(shape = (input_dim1,))
    input_layer2 = Input(shape = (input_dim2,))
    input_layer3 = Input(shape = (input_dim3,))
    input_layer4 = Input(shape = (input_dim4,))
    input_layers = [input_layer1, input_layer2, input_layer3, input_layer4]

    e1 = Dense(15)(input_layer1)
    #e1 = Embedding(output_dim=30, input_dim=num_items, input_length=2)(input_layer2)
    #f1 = Flatten()(e1)
    # f1 = BatchNormalization()(e1)
    f1 = Activation('relu')(e1)
    f1 = Dense(50)(f1)
    f1 = Activation('relu')(f1)
    f1 = Dense(100)(f1)
    f1 = Activation('relu')(f1)
    # f1 = Dropout(0.2)(f1)
    
    e2 = Dense(30)(input_layer2)
    f2 = Activation('relu')(e2)
    e2 = Dense(80)(input_layer2)
    f2 = Activation('relu')(e2)
    f2 = Dense(200)(f2)
    f2 = Activation('relu')(f2)
    # f2 = Dropout(0.2)(f2)
    f2 = Dense(15)(f2)
    f2 = Activation('relu')(f2)
    # f2 = Dense(5)(f2)
    # f2 = Activation('tanh')(f2)

    d3 = Dense(50)(input_layer3)
    d3 = Activation('relu')(d3)
    d3 = Dense(90)(input_layer3)
    d3 = Activation('relu')(d3)
    d3 = Dense(500)(d3)
    # d3 = Dropout(0.20)(d3)
    d3 = Activation('relu')(d3)
    d3 = Dense(10)(d3)
    d3 = Activation('relu')(d3)
    # d3 = Dense(5)(d3)
    # #d3 = BatchNormalization()(d3)
    # d3 = Activation('tanh')(d3)

    c1 = Dense(100)(input_layer4)
    c1 = Activation('relu')(c1)
    c1 = Dense(500)(c1)
    c1 = Activation('relu')(c1)
    # c1 = Dropout(0.15)(c1)
    # c1 = Dense(50)(c1)
    # c1 = Activation('relu')(c1)

    do = keras.layers.concatenate([f1, f2, d3, c1], axis=-1)
    do = Dense(100)(do)
    do = Activation('relu')(do)

    
    #do = keras.layers.concatenate([do, c1], axis=-1)
    do = Dense(1)(do)
    out_layer = Activation('relu')(do)
    rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model = Model(input = input_layers, output = out_layer)
    model.compile(loss = 'mae', optimizer='adam', metrics = [acc_loss])
    return model

def get_model_layer_weights(model, out_layer_index, input_data):
    in_layers = model.input
    in_layers.append((K.learning_phase()))
    out_layers = model.layers[-3]
    func = K.function(in_layers, [out_layers.output])
    outs = func(input_data)
    outs = np.array(outs)
    #print (outs.shape)
    outs = outs.reshape((outs.shape[1], outs.shape[2]))
    return outs
    
if __name__ == "__main__":
    ####################################################################################
    model = fcnn(6,25,10,79)
    print ("Model Loaded ............")
    ####################################################################################################################################
    
    CHUNKSIZE = 120000
    TEST_CHUNKSIZE = 1200000
    NUM_OF_ITERATIONS = 100
    BATCH_SIZE = 1000
    TRAIN_SIZE = 92000000
    STOP_DL_TRAIN_CHUNK_COUNT = 100
    if sys.argv[-1] == 'dev':
        TEST_PATH = 'data/test.csv'
        TRAIN_PATH = 'data/train.csv'
    else: 
        TEST_PATH = '../input/test.csv'
        TRAIN_PATH = '../input/train.csv'
    
    ################## XGB Parameters ##########################################
    param = {
        'max_depth': 10,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        "objective": "reg:linear", 
        "booster":"gblinear",  # error evaluation for multiclass training
        }  # the number of classes that exist in this datset
    num_round = 20  # the number of training iterations
    ################## XGB Parameters ##########################################
    
    # vis_data = pd.read_csv(TRAIN_PATH, nrows = 100)
    # vis_cat_data = prepare_cat_data(vis_data)
    # x_vis_data, y_vis_data = prepare_data(vis_data)
    # x_vis_data.append(vis_cat_data)
    train_all = pd.read_csv(TRAIN_PATH, skiprows = range(1, 101688779))
    chunk_count=0
    # for train in pd.read_csv(TRAIN_PATH, chunksize=CHUNKSIZE):#, skiprows = range(1, 125397041))
    for i_t in range(0, TRAIN_SIZE, CHUNKSIZE):
        train = train_all.loc[i_t:i_t+CHUNKSIZE]
        print (i_t, i_t+CHUNKSIZE, train.shape, train_all.shape,1111111111111111111111111)
        chunk_count+=1
        
    
        train.loc[train.unit_sales < 0, 'unit_sales'] = 0
        train['date'] = pd.to_datetime(train['date'])
        
        train['year'] = train['date'].dt.year
        train['sin_month'] = np.sin(2*np.pi*train['date'].dt.month/12)
        train['sin_day'] = np.sin(2*np.pi*train['date'].dt.day/31)
        train['sin_dayofweek'] = np.sin(2*np.pi*train['date'].dt.dayofweek/7)
        train['cos_month'] = np.cos(2*np.pi*train['date'].dt.month/12)
        train['cos_day'] = np.cos(2*np.pi*train['date'].dt.day/31)
        train['cos_dayofweek'] = np.cos(2*np.pi*train['date'].dt.dayofweek/7)

        train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion
        
        item_nbr_bin = train['item_nbr'].apply(lambda x: [i for i in bin(x)[2:]])
        item_nbr_bin = keras.preprocessing.sequence.pad_sequences(item_nbr_bin, maxlen=25, dtype='int32', padding='pre', truncating='pre', value=0.)
        
        store_nbr_bin = train['store_nbr'].apply(lambda x: [i for i in bin(x)[2:]])
        store_nbr_bin = keras.preprocessing.sequence.pad_sequences(store_nbr_bin, maxlen=10, dtype='int32', padding='pre', truncating='pre', value=0.)
        categoricals_train = prepare_cat_data(train)
        
        
        print ('train, test Data loaded ...')
        print ('train shape: ', train.shape)
        #test_all = pd.merge(test, train, how = 'left', on = ['item_nbr','store_nbr', 'dayofweek', 'month', 'date'])
        
        itr = 0
        for ri in random.sample(range(100, CHUNKSIZE-10000), NUM_OF_ITERATIONS):
            itr += 1
            # [id,date,store_nbr,item_nbr,unit_sales,onpromotion]
            # print (categoricals_train[ri: ri+BATCH_SIZE].shape, 222222222222222)
            x_train = [np.array(train.drop(['id','date','unit_sales','onpromotion', 'item_nbr','store_nbr', 'year'], axis=1)[ri:ri+BATCH_SIZE]), item_nbr_bin[ri:ri+BATCH_SIZE], store_nbr_bin[ri:ri+BATCH_SIZE], categoricals_train[ri: ri+BATCH_SIZE]]
            y_train = np.array(train['unit_sales'][ri:ri+BATCH_SIZE])
            y_valid = np.array(train['unit_sales'][-1000:])
            x_valid = [np.array(train.drop(['id','date','unit_sales','onpromotion', 'item_nbr','store_nbr', 'year'], axis=1)[-1000:]), item_nbr_bin[-1000:], store_nbr_bin[-1000:], categoricals_train[-1000:]]
            
            if chunk_count < STOP_DL_TRAIN_CHUNK_COUNT:
                train_logs = model.train_on_batch(x_train, y_train)
                valid_logs = model.test_on_batch( x_valid, y_valid)
            if (itr%20==0):
                print ('====================== CHUNK NO: ', chunk_count, 'ITERAITON: ', itr, ' ================================')
                if chunk_count < STOP_DL_TRAIN_CHUNK_COUNT:
                    print ('Train log: ', train_logs, '\nTest log: ', valid_logs)
                if chunk_count > STOP_DL_TRAIN_CHUNK_COUNT:
                    dl_weights_train = get_model_layer_weights(model, -3, x_train)
                    dl_weights_valid = get_model_layer_weights(model, -3, x_valid)
                    x_xgb = np.concatenate((np.array(train.drop(['id','date','unit_sales','onpromotion', 'item_nbr','store_nbr', 'year'], axis=1)[ri:ri+BATCH_SIZE]), item_nbr_bin[ri:ri+BATCH_SIZE], store_nbr_bin[ri:ri+BATCH_SIZE], categoricals_train[ri: ri+BATCH_SIZE], dl_weights_train), axis=1)
                    #print (x_xgb.shape, 11111111111111)
                    x_xgb = xgb.DMatrix(x_xgb, label = y_train)
                    bst = xgb.train(param, x_xgb, num_round)
                    x_xgb_valid = np.concatenate((np.array(train.drop(['id','date','unit_sales','onpromotion', 'item_nbr','store_nbr', 'year'], axis=1)[-1000:]), item_nbr_bin[-1000:], store_nbr_bin[-1000:], categoricals_train[-1000:], dl_weights_valid), axis=1)
                    x_xgb_valid = xgb.DMatrix(x_xgb_valid, label = y_valid)
                    y_xgv_valid_pred = bst.predict(x_xgb_valid)
                    print ("XGB VALIDATION LOSS: %f ---->>" %(nrms(y_valid, y_xgv_valid_pred)))
            
                y_valid_pred = model.predict(x_valid)
                # print ("standard deviation : %f , Mean : %f of training chunk ---->>" %(y_train.std(), y_train.mean()))
                print ("standard deviation : %f , Mean : %f of Validating chunk ---->>" %(y_valid.std(), y_valid.mean()))
                # vis_predict_data = model.predict(x_vis_data)
                # vis2(y_vis_data, vis_predict_data)
                print ("standard deviation : %f , Mean : %f of Predicted Validating chunk ---->>" %(y_valid_pred.std(), y_valid_pred.mean()))
        if chunk_count >1000: break
    
    df_test = pd.DataFrame(columns=["id",'unit_sales'])
    for test in pd.read_csv(TEST_PATH, chunksize = TEST_CHUNKSIZE):#, nrows= 10)
    #save_model(model, "model.h5")
        test['date'] = pd.to_datetime(test['date'])
        test['year'] = test['date'].dt.year
        test['sin_month'] = np.sin(2*np.pi*test['date'].dt.month/12)    
        test['sin_day'] = np.sin(2*np.pi*test['date'].dt.day/31)
        test['sin_dayofweek'] = np.sin(2*np.pi*test['date'].dt.dayofweek/7)
        test['cos_month'] = np.cos(2*np.pi*test['date'].dt.month/12)
        test['cos_day'] = np.cos(2*np.pi*test['date'].dt.day/31)
        test['cos_dayofweek'] = np.cos(2*np.pi*test['date'].dt.dayofweek/7)
    
        test_item_nbr_bin = test['item_nbr'].apply(lambda x: [i for i in bin(x)[2:]])
        test_item_nbr_bin = keras.preprocessing.sequence.pad_sequences(test_item_nbr_bin, maxlen=25, dtype='int32', padding='pre', truncating='pre', value=0.)
        
        test_store_nbr_bin = test['store_nbr'].apply(lambda x: [i for i in bin(x)[2:]])
        test_store_nbr_bin = keras.preprocessing.sequence.pad_sequences(test_store_nbr_bin, maxlen=10, dtype='int32', padding='pre', truncating='pre', value=0.)
        # print ('test data prepared ...111')
        categoricals_test = prepare_cat_data(test)
        print ('test data prepared ...')
        y_pred = model.predict([np.array(test.drop(['id','date', 'onpromotion', 'item_nbr','store_nbr', 'year'], axis=1)),test_item_nbr_bin, test_store_nbr_bin, categoricals_test])
        print ('results predicted ...')
        sub_dic = {'id': test['id'], 'unit_sales': y_pred.reshape(len(test))}
        df = pd.DataFrame(sub_dic, columns=["id",'unit_sales'])
        df['unit_sales'] = df['unit_sales'].apply(pd.np.expm1)
        
        print ("standard deviation : %f , Mean : %f of Test chunk" %(df['unit_sales'].std(), df['unit_sales'].mean()))
        df_test = pd.concat([df, df_test])
        print ('test shape till now : ', df_test.shape)
    # print (df.head())
    df_test.to_csv('keras_submission.csv', index=False)
