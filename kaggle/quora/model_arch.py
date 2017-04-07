from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LSTM, Lambda, Merge, Reshape, GRU, Input, merge, Flatten
from keras import regularizers
#from keras.optimizers import SGD
#from keras.optimizers import SGD, RMSprop
#from keras.regularizers import l2, activity_l2, l1
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import optimizers

import numpy as np

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def siamese(max_sent_len, wordvec_dim,gru_output_dim, output_dim):
    #gru_sent_fwd = LSTM(gru_output_dim, return_sequences=False, go_backwards = False, activation='relu', inner_activation='hard_sigmoid', input_shape = (max_sent_len,wordvec_dim))
    #gru_sent_bwd = LSTM(gru_output_dim, return_sequences=False, go_backwards = True, activation='relu', inner_activation='hard_sigmoid', input_shape = (max_sent_len,wordvec_dim))
    #input_layers = []
    gru_f, gru_b, gru_sent_fwd, gru_sent_bwd= [],[],[],[]
    input_layers = [Input(shape = (max_sent_len, wordvec_dim)) for i in xrange(2)]
    #input_2 = Input(shape = (max_sent_len, wordvec_dim))
    siamese_sub_parts = []
    for i in xrange(2):
        gru_sent_fwd.append(GRU(gru_output_dim, return_sequences=False, go_backwards = False, activation='tanh', inner_activation='hard_sigmoid', input_shape = (max_sent_len,wordvec_dim)))
        gru_sent_bwd.append(GRU(gru_output_dim, return_sequences=False, go_backwards = True, activation='tanh', inner_activation='hard_sigmoid', input_shape = (max_sent_len,wordvec_dim)))
	gru_f.append(gru_sent_fwd[i](input_layers[i]))
	gru_b.append(gru_sent_bwd[i](input_layers[i]))
	#gru_f[i] = BatchNormalization()(gru_f[i])
	gru_f[i] = Activation('tanh')(gru_f[i])
	#gru_b[i] = BatchNormalization()(gru_b[i])
        gru_b[i] = Activation('tanh')(gru_b[i])
	gru_merge = merge([gru_f[i], gru_b[i]], mode = 'concat', concat_axis = -1)
        gru_merge = Dense(20)(gru_merge)
        #gru_merge = Dropout(0.25)(gru_merge)
        #gru_merge = Dense(100)(gru_merge)
        siamese_sub_parts.append(gru_merge)

    #distance = Lambda(euclidean_distance, output_shape = eucl_dist_output_shape)(siamese_sub_parts)
    #model = Dense(output_dim)(distance)

    model = merge(siamese_sub_parts, mode = 'concat', concat_axis = -1)
    model = Dense(output_dim)(model)
    
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    #model = BatchNormalization()(model)
    out_layer = Activation('softmax')(model)
    model = Model(input = input_layers, output = out_layer)
    
    model.compile(loss = 'categorical_crossentropy', optimizer='adagrad', metrics = ['accuracy'])
    
    return model

def dense_test(max_sent_len, wordvec_dim,gru_output_dim, output_dim):
    input_layers = [Input(shape = (max_sent_len, wordvec_dim)) for i in xrange(2)]
    i1 = Flatten()(input_layers[0])
    i2 = Flatten()(input_layers[1])
    d1 = Dense(32, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(i1)
    d2 = Dense(32, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(i2)
    m = merge([d1,d2], mode = 'concat', concat_axis = -1)
    d3 = Dense(2, kernel_initializer='random_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01))(m)
    d3 = BatchNormalization()(d3)
    out_layer = Activation('softmax')(d3)
    model = Model(input = input_layers, output = out_layer)
    model.compile(loss = 'categorical_crossentropy', optimizer='adagrad', metrics = ['accuracy'])
    return model
