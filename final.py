#import kagglegym
import sys
import pandas as pd
import numpy as np
from keras.models import  Model
from random import randint
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, LSTM, Lambda, Merge, Reshape, GRU, Input, merge, Dense, Activation, Merge, Reshape, Dropout, Flatten

train = pd.read_hdf('../kaggle_data/train.h5')


train_avg = train.y.mean()
print('y average value in training set:', train_avg)

# fields_all = train.keys()
# fields_needed = filter(lambda x: isinstance(train[x][0], np.float32), fields_all)
# print fields_needed
# fields_needed = ['fundamental_18', 'fundamental_2', 'fundamental_11', 'fundamental_16', 'fundamental_3', 'fundamental_9', 'technical_16', 'technical_19', 'fundamental_8', 'technical_25']
max_y = max(train.y)
min_y = min(train.y)

# fields_all = train.keys()
# fields_needed = list(filter(lambda x: isinstance(train[x][0], np.float32), fields_all))
# fields_needed = fields_needed[:5]
fields_needed = ['fundamental_18', 'fundamental_2', 'fundamental_11', 'fundamental_16']
# fields_needed = ['fundamental_18', 'fundamental_2', 'fundamental_11', 'fundamental_16', 'fundamental_3', 'fundamental_9', 'technical_16', 'technical_19', 'fundamental_8', 'technical_25']
# print (fields_needed,111111111111111111111111)


out_field = ['y']
y_train= train[out_field]

try:
    fields_needed.remove('y')
    # fields.remove('id')
    # fields.remove('timestamp')
except: pass

df = train[fields_needed]
mean_values = df.mean(axis=0)
df.fillna(mean_values, inplace=True)
df = (df - df.mean()) / (df.max() - df.min())
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

    model = Reshape((16,16))(model)
    model = Convolution1D(32, 4, border_mode = 'same')(model)
    model = MaxPooling1D(2)(model)
    model = Flatten()(model)
    model = Activation('relu')(model)

    # model = Dense(128)(model)
    model = Dense(16)(model)
    model = Dense(32)(model)
    model = Dense(64)(model)
    model = Activation('sigmoid')(model)

    model = Reshape((8,8))(model)
    model = Convolution1D(8, 4, border_mode = 'same')(model)
    model = MaxPooling1D(2)(model)
    model = Flatten()(model)
    model = Activation('relu')(model)

    model = Dense(16)(model)
    model = Dense(1)(model)
    out_layer = Activation('linear')(model)
    model = Model(input=input_layer, output=out_layer)
    model.compile(loss='mae', optimizer='sgd', metrics=['mae'])
    return model

def compile_model_2d(input_shape):
    input_1 = Input(shape = (input_shape))
    input_layer = [input_1]
    
    model = Convolution2D(10,3,3)(input_1)
    model = MaxPooling2D((2,2), strides=(1,1), border_mode='valid')(model)
    model = Activation('relu')(model)
    print filter(lambda x: not x.endswith('_'),dir(model)),222222222
    print model.get_shape,22222222222222

    model = Reshape((10,7))(model)

    model_f = LSTM(32, return_sequences=False, go_backwards = False, activation='tanh', inner_activation='hard_sigmoid')
    model_b = LSTM(32, return_sequences=False, go_backwards = True, activation='tanh', inner_activation='hard_sigmoid')

    model_f = model_f(model)
    model_f = Activation('relu')(model_f)

    model_b = model_f(model)
    model_b = Activation('relu')(model_b)

    model_merge = Merge([model_f, model_b], mode='concat', concat_axix=-1)

    model_merge = Dense(64)(model_merge)
    model_merge = Dense(16)(model_merge)
    model_merge = Dense(1)(model_merge)
    model_merge = Activation('linear')(model_merge)


train_size = 100000

def batch_gen(batch_size):
    while(True):
        i = 0
        i = randint(0, len(df)-train_size)
        x_batch = np.array(x_train[i:i+batch_size-1])
        y_batch = np.array(y_train[i:i+batch_size-1])
        # i = i+batch_size
            
        yield x_batch, y_batch

def batch_gen_2d(batch_size):
    while True:
        i = randint(10, len(df)-train_size-10)
        y_batch = np.array(y_train[i:i+batch_size-1])
        x_batch = np.array([[y_train[k-j:k+1] for j in range(10,0,-1)] for k in range(i, i+batch_size-1)])
        yield x_batch, y_batch

# input_shape = x_train.shape[1]
batch_size = 500
input_shape = (1,10,x_train.shape[1])
print("initializing model...")
model = compile_model_2d(input_shape)
print("fitting model...")
fit = model.fit_generator(generator=batch_gen_2d(500, train_input=True), nb_epoch=100, samples_per_epoch=train_size)