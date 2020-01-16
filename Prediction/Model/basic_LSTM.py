from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import math
import pickle
from random import shuffle

# path = get_file(
#     'nietzsche.txt',
#     origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
# with io.open(path, encoding='utf-8') as f:
#     text = f.read().lower()
# print('corpus length:', len(text))

# chars = sorted(list(set(text)))
# print('total chars:', len(chars))
# char_indices = dict((c, i) for i, c in enumerate(chars))
# indices_char = dict((i, c) for i, c in enumerate(chars))

# # cut the text in semi-redundant sequences of maxlen characters
# maxlen = 40
# step = 3
# sentences = []
# next_chars = []
# for i in range(0, len(text) - maxlen, step):
#     sentences.append(text[i: i + maxlen])
#     next_chars.append(text[i + maxlen])
# print('nb sequences:', len(sentences))

# print('Vectorization...')
# x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
# y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
# for i, sentence in enumerate(sentences):
#     for t, char in enumerate(sentence):
#         x[i, t, char_indices[char]] = 1
#     y[i, char_indices[next_chars[i]]] = 1

dataset = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/pre-process/1464_day_op_count.pkl'
day_info = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/pre-process/1464_day_info.pkl'
index_info = '/Users/nelson/Academia/Research/Work/SmartHEMS/Prediction/Data Source/pre-process/1464_indexing.pkl'

dev_num = 1


lstm_hidden_dim = 32
nn_hidden_dim = 32


roll_number = 0



#load x and y

with open(dataset, 'rb') as f:
        datas = np.array(pickle.load(f))

with open(day_info, 'rb') as f:
        days = np.array(pickle.load(f))

with open(index_info, 'rb') as f:
        indexs = np.array(pickle.load(f))


# x = np.zeros(roll_number,7*8) # 3, (0,0,0,0,0,0,1)
# y = np.zeros(roll_number)


# Tune inputs into X and Y
datas = datas[dev_num, :]

X = list()
Y = list()
for item in indexs:
    x = list()

    for i in range(7):
        x.append(datas[item[i]])
    Y.append(datas[item[-1]])
    X.append(x)

X = np.array(X)
Y = np.array(Y)

train_idx = math.floor(len(X) * 0.8)

X_training = X[:train_idx, :]
X_testing = X[train_idx:, :]

Y_training = Y[:train_idx]
Y_testing = Y[train_idx:]


#lahiru's code
X_train=np.reshape(X_training,(X_training.shape[0],1,X_training.shape[1]))
Y_train=np.reshape(Y_training,(Y_training.shape[0],1))

X_test = np.reshape(X_testing, (X_testing.shape[0], 1, X_testing.shape[1]))
Y_test = np.reshape(Y_testing, (Y_testing.shape[0], 1))

X_train = np.true_divide(X_train, 24)
Y_train = np.true_divide(Y_train, 24)

X_test = np.true_divide(X_test, 24)
Y_test = np.true_divide(Y_test, 24)

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add( LSTM(32, input_shape=(1, 7) ) )
model.add( Dense(32, activation='relu') )
model.add( Dense(1,  activation='relu'))
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=optimizer)

model.fit(X_train, Y_train,
          batch_size=32,
          epochs=60)

model.evaluate(x = X_test, y = Y_test)



preds = model.predict(X_test, verbose=0)
for i in range(len(preds)):
   print('Prediction: ', preds[i]*24)
   print('Groud Truth: ', Y_test[i]*24)
