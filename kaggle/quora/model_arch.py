from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LSTM, Lambda, Merge, Reshape, GRU, Input, merge
from keras.optimizers import SGD
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2, activity_l2, l1
from keras.layers.normalization import BatchNormalization
import numpy as np

def siamese(max_sent_len, wordvec_dim,gru_output_dim, output_dim):
    gru_sent_fwd = LSTM(gru_output_dim, return_sequences=False, go_backwards = False, activation='tanh', inner_activation='hard_sigmoid', input_shape = (max_sent_len,wordvec_dim))
    gru_sent_bwd = LSTM(gru_output_dim, return_sequences=False, go_backwards = True, activation='tanh', inner_activation='hard_sigmoid', input_shape = (max_sent_len,wordvec_dim))
    #input_layers = []
    
    input_layers = [Input(shape = (max_sent_len, wordvec_dim)) for i in xrange(2)]
    #input_2 = Input(shape = (max_sent_len, wordvec_dim))
    siamese_sub_parts = []
    for i in xrange(2):
	gru_f = gru_sent_fwd(input_layers[i])
	gru_b = gru_sent_bwd(input_layers[i])
	gru_f = BatchNormalization()(gru_f)
	gru_f = Activation('tanh')(gru_f)
	gru_b = BatchNormalization()(gru_b)
        gru_b = Activation('tanh')(gru_b)
	gru_merge = merge([gru_f, gru_b], mode = 'concat', concat_axis = -1)
        gru_merge = Dense(200)(gru_merge)
        gru_merge = Dropout(0.25)(gru_merge)
        gru_merge = Dense(100)(gru_merge)
        siamese_sub_parts.append(gru_merge)

    model = merge(siamese_sub_parts, mode = 'concat', concat_axis = -1)
    model = Dense(output_dim)(model)
    out_layer = Activation('sigmoid')(model)
    model = Model(input = input_layers, output = out_layer)
    
    model.compile(loss = 'categorical_crossentropy', optimizer='sgd', metrics = ['accuracy'])
    
    return model

