#import kagglegym
import sys
import pandas as pd
import numpy as np
from keras.models import  Model
from random import randint
from keras.layers import Convolution2D, MaxPooling2D, LSTM, Lambda, Merge, Reshape, GRU, Input, merge, Dense, Activation, Merge, Reshape, Dropout


train = pd.read_hdf('../kaggle_data/train.h5')


train_avg = train.y.mean()
print('y average value in training set:', train_avg)

fields_all = train.keys()
fields_needed = filter(lambda x: isinstance(train[x][0], np.float32), fields_all)
print fields_needed
df = train[fields_needed]
mean_values = df.mean(axis=0)
df.fillna(mean_values, inplace=True)

out_field = ['y']
y_train= df[out_field]

try:
    fields_needed.remove('y')
    # fields.remove('id')
    # fields.remove('timestamp')
except: pass

x_train= df[fields_needed]

# mean_values = x_train.mean(axis=0) 

print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)

def compile_model(input_shape):
    # gru_fw = GRU(gru_word_output_dim, return_sequences=False, go_backwards = False, activation='tanh', inner_activation='hard_sigmoid', input_shape = (80))

    input_1 = Input(shape = (input_shape,))
    input_layer = [input_1]
    # input_1 = Reshape((input_shape))(input_1)
    model = Dense(256)(input_1)
    model = Activation('softmax')(model)
    model = Dense(128)(model)
    model = Dense(64)(model)
    model = Dense(1)(model)
    out_layer = Activation('linear')(model)
    model = Model(input=input_layer, output=out_layer)
    model.compile(loss='mae', optimizer='sgd', metrics=['mae'])
    return model

train_size = 100000

def batch_generator(batch_size, train_input=True):
    while(True):
        i = randint(0, len(df)-train_size)
        x_batch = np.array(x_train[i:i+batch_size-1])
        y_batch = np.array(y_train[i:i+batch_size-1])
        i = i+batch_size
            
        yield x_batch, y_batch

input_shape = x_train.shape[1]

print("initializing model...")
model = compile_model(input_shape)
print("fitting model...")
fit = model.fit_generator(generator=batch_generator(500, train_input=True), nb_epoch=100, samples_per_epoch=train_size)

