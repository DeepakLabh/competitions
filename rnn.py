#import kagglegym
import sys
import pandas as pd
import numpy as np
from keras.models import  Model
from random import randint
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, LSTM, Lambda, Merge, Reshape, GRU, Input, merge, Dense, Activation, Merge, Reshape, Dropout, Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam, RMSprop

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
fields_needed = ['fundamental_18', 'fundamental_2', 'fundamental_11', 'fundamental_16'] #top correlated
fields_needed =["technical_20","technical_30","technical_21", "technical_2","technical_17","technical_19","technical_11","technical_12"]
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

def pearson_r(y_true, y_pred):
    return (np.corrcoef(list(y_true), list(y_pred))[0, 1])




def compile_model_2d(input_shape):
    input_1 = Input(shape = (input_shape))
    input_layer = [input_1]
    # model = Convolution2D(10,3,3, border_mode='same')(input_1)
    # model = MaxPooling2D((2,2), strides=(1,1), border_mode='same')(model)
    # model = Activation('relu')(model)

    # model = Reshape((10,10))(model)

    model_f = LSTM(32, return_sequences=False, go_backwards = False, activation='tanh', inner_activation='hard_sigmoid')
    model_b = LSTM(32, return_sequences=False, go_backwards = True, activation='tanh', inner_activation='hard_sigmoid')

    model_f = model_f(input_1)
    model_f = Activation('relu')(model_f)

    model_b = model_b(input_1)
    model_b = Activation('relu')(model_b)

    model_merge = merge([model_f, model_b], mode='concat', concat_axis=-1)

    model_merge = Dense(64)(model_merge)
    # model_merge = Dense(16)(model_merge)
    model_merge = Dense(1)(model_merge)
    out_layer = Activation('linear')(model_merge)
    model_final = Model(input=input_layer, output=out_layer)
    rms_prop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model_final.compile(loss='mean_squared_error', optimizer=rms_prop, metrics=['pearson_r'])
    return model_final


train_size = 100000



def batch_gen_2d(batch_size):
    while True:
        i = randint(10, len(df)-train_size-10)
        y_batch = np.array(y_train[i:i+batch_size])
        x_batch = ([x_train[k-9:k+1] for k in range(i, i+batch_size)])
        # x_batch = ([[x_train[k-j:k+1] for j in range(10,0,-1)] for k in range(i, i+batch_size-1)])
        # print (len(x_batch), len(x_batch[0]), len(x_batch[2])),3333333333333333
        x_batch = map(lambda x: np.reshape(np.array(x), (10, x_train.shape[1])),x_batch)
        # print np.array([x_batch[0],x_batch[1]]).shape,4444444444444444
        x_batch = np.array(x_batch)

        yield x_batch, y_batch
        # break

# input_shape = x_train.shape[1]
batch_size = 500
input_shape = (10,x_train.shape[1])
print("initializing model...")
model = compile_model_2d(input_shape)
print("fitting model...")
fit = model.fit_generator(generator=batch_gen_2d(500), nb_epoch=100, samples_per_epoch=train_size)