import pandas as pd
import sys
import numpy as np


#train = pd.read_csv('data/train_2016_v2.csv')
#props = pd.read_csv('data/properties_2016.csv')
#
#mean_props = props.mean()
#props.fillna(mean_props)
##props_train = props.loc[props['parcelid'].isin(train['parcelid'])]
#train_all_fields = pd.merge(train, props, how='inner', on=['parcelid'])
#train_all_fields.to_csv('all_training_data.csv')


all_data = pd.read_csv('data/zillo.csv')

# View the keys (details)
key_details = all_data.columns.to_series().groupby(all_data.dtypes).groups
object_keys = key_details[np.dtype('O')]
print (key_details,'\n Object Keys are: ',object_keys)

## Write a Model
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Merge, merge, Reshape, Input
from keras.optimizers import RMSprop, SGD
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import optimizers
import keras

print ('loading libraries done ...')
# define the deep learning model
def model(input_dim):
    input_layers = Input(shape = (input_dim,))
    d1 = Dense(30)(input_layers)
    d1 = BatchNormalization()(d1)
    d2 = Activation('tanh')(d1)
    d1 = Dense(10)(input_layers)
    d1 = BatchNormalization()(d1)
    d2 = Activation('sigmoid')(d1)
    d2 = Dense(1)(d2)
    d2 = BatchNormalization()(d2)
    out_layer = Activation('linear')(d2)
    rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model = Model(input = input_layers, output = out_layer)
    model.compile(loss = 'mae', optimizer='sgd')
    return model

# Create the training data.
print ('replacing all data with corresponding mean ...')
#all_data = all_data.fillna(np.zeros(all_data.shape))
all_data = all_data.fillna(all_data.mean())
all_data['date'] = map(lambda x:int(x.split('-')[2]), all_data['transactiondate'])
all_data['month'] = map(lambda x:int(x.split('-')[1]), all_data['transactiondate'])
all_data['year'] = map(lambda x:int(x.split('-')[0]), all_data['transactiondate'])
try:y = all_data['logerror']
except: pass
try:all_data = all_data.drop(list(object_keys), axis=1)
except: pass

# Load Model
mod = model(all_data.shape[1])
print ('model loaded', '...')
print ('y_max:', max(y), 'y_min:', min(y))
train_size = int(0.9*len(all_data))
all_data = all_data.astype(np.float32)
x_train = np.array(all_data[:train_size])
y_train =  np.array(y[:train_size])
x_val = np.array(all_data[train_size:])
y_val = np.array(y[train_size:])

print (x_train.shape, y_train.shape, x_val.shape, y_val.shape)
mod.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=200, batch_size=8, verbose=1)
print ('training done..' )


