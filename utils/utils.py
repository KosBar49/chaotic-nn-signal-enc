from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np


def dec_to_byte(a):
    bnr = bin(a).replace('0b','')
    x = bnr[::-1] #this reverses an array
    while len(x) < 8:
        x += '0'
    bnr = x[::-1]
    return bnr

def init_model():
    def hardlim(x):
        return K.cast(K.greater_equal(x,0), K.floatx())
    
    model = Sequential()
    model.add(Dense(8, input_dim = 8, activation = hardlim))
    model.summary()
    return model

def init_weights(byte_arr):
    weights = np.array([8*[0]])
    
    for i in range(0, 7):
        weights = np.append(weights, [8*[0]], axis=0)
        
    bias = 8*[0]
    
    for i in range(0, 8):
        for j in range(0, 8):
            if byte_arr[i] == 0:
                bias[i] = -0.5
                if j == i:
                    weights[j][i] = 1
            if byte_arr[i] == 1:
                bias[i] = 0.5
                if j == i:
                    weights[j][i] = -1
                    
    return (weights, bias)

def init_chaotic_seq(u, x1, l):
    x = np.array([x1])
    for i in range(0, l - 1):
        x = np.append(x, u*x[i]*(1 - x[i]))
    x = np.ceil(( x - np.min(x)) / np.max(x)  * 255)
    x  = [int(i) for i in x]
    return x
