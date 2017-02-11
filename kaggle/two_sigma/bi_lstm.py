# import kagglegym
import sys
import pandas as pd
import numpy as np
import keras
from keras.models import  Model
from random import randint
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D, LSTM, Lambda, Merge, Reshape, GRU, Input, merge, Dense, Activation, Merge, Reshape, Dropout, Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam, RMSprop
import tensorflow as tf

print ('tensorflow version: ', tf.__version__)
print ('numpy version: ', np.__version__)
print ('Keras version: ', keras.__version__)


############################################################
# The "environment" is our interface for code competitions
# env = kagglegym.make()

# We get our initial observation by calling "reset"
# observation = env.reset()

# Note that the first observation we get has a "train" dataframe
# print("Train has {} rows".format(len(observation.train)))

# The "target" dataframe is a template for what we need to predict:
# print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))


############################################################

train = pd.read_hdf('../kaggle_data/train.h5')
# train = observation.train

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
fields_needed =["technical_20","technical_30", "technical_19","timestamp"]
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
    # model_f = Activation('softmax')(model_f)

    # model_b = model_b(input_1)
    # model_b = Activation('relu')(model_b)

    # model = merge([model_f, model_b], mode='concat', concat_axis=-1)

    model = Dense(1)(model_f)
    # model = Dense(16)(model)
    # model = Dense(1)(model)
    out_layer = Activation('sigmoid')(model)
    model_final = Model(input=input_layer, output=out_layer)
    rms_prop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model_final.compile(loss='mean_squared_error', optimizer='adam')
    # model_final.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model_final



train_size = 100000



def batch_gen_2d(batch_size):
    while True:
        i = randint(input_shape[0], len(df)-train_size-input_shape[0])
        y_batch = np.array(y_train[i:i+batch_size])
        x_batch = ([x_train[k-input_shape[0]+1:k+1] for k in range(i, i+batch_size)]) #  Create 2-D input
        # x_batch = np.array(x_batch)
        # print (len(x_batch),33333333333, len(x_batch[0]))
        # x_batch = ([[x_train[k-j:k+1] for j in range(10,0,-1)] for k in range(i, i+batch_size-1)])
        # print (len(x_batch), len(x_batch[0]), len(x_batch[0].keys())),3333333333333333
        x_batch = map(lambda x: np.reshape(np.array(x), (input_shape[0], x_train.shape[1])),x_batch) # 10 is nu of recurrent layers
        # print (x_batch.shape,4444444444444444)
        
        x_batch = np.array(list(x_batch))
        

        yield (x_batch, y_batch)
        # break

# input_shape = x_train.shape[1]
batch_size = 500
input_shape = (20,x_train.shape[1])
print("initializing model...")
model = compile_model_2d(input_shape)
print("fitting model...")
fit = model.fit_generator(generator=batch_gen_2d(500), nb_epoch=30, validation_data = batch_gen_2d(5000), nb_val_samples = 5000, samples_per_epoch=train_size)
# print (observation.target,333333333333333333333, observation.features.keys(),222222222)
###########################################  Testing from kagglegym api ################################
# while True:
#     # # target = observation.target
#     x_test = observation.features[fields_needed].fillna(mean_values, inplace = True)
#     x_pad = [[0 for i in range(0, x_test.shape[1])] for j in range(0, 10)]
#     x_pad = np.array(x_pad)
#     x_test = np.concatenate((x_pad,x_test), axis = 0)
#     # print (np.array(x_pad).shape, x_test.shape)
#     # x_test = x_pad+list(x_test)
#     # print (len(x_test), len(x_test[0]))
#     x_test = [x_test[i-10: i ] for i in range(10, len(x_test))]
#     # print (len(x_test), len(x_test[0]), len(x_test[0][0]))
#     x_test = map(lambda x: np.reshape(np.array(x), (10, x_train.shape[1])), x_test)
#     x_test = np.array(list(x_test))
    
#     # # print (x_test.shape)
#     # # target.y = model.predict(x_test)
#     target = observation.target
#     target.y = model.predict(x_test)
#     target.y[observation.target.y>max_y] = max_y
#     target.y[observation.target.y<min_y] = min_y
#     # observation.target.y = [y_mode[0] for i in range(0, x_train.shape[0])]
#     timestamp = observation.features['timestamp'][0]
#     observation, reward, done, info = env.step(target)
#     if timestamp%100 == 0:
#         print("Timestamp #{}".format(timestamp), reward)
#         print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
#         print(target.y.head())
#     if done:
#         print ('finished ###')
#         print("Public score: {}".format(info["public_score"]))
#         break
