import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from keras import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.activations import sigmoid, linear
from keras.losses import mean_squared_error

from working_data import get_data
from cross_validation import cross_validate

data = pd.read_csv("bdata.20130222.mhci.txt", '\t')
featu_enc, x, y = get_data(data, dynamic=False)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
print(x.shape)

# MLP
model = Sequential()
model.add(Dense(input_shape=(180,), units=80, activation=sigmoid))
model.add(Dense(80, activation=sigmoid))
model.add(Dense(1, activation=linear))
model.compile(optimizer="adam", loss=mean_squared_error, metrics=['mse'])


model2 = Sequential()
model2.add(Dense(input_shape=(180,), units=80, activation=sigmoid))
model2.add(Dense(80, activation=sigmoid))
model2.add(Dense(80, activation=sigmoid))
model2.add(Dense(1, activation=linear))
model2.compile(optimizer="adam", loss=mean_squared_error, metrics=['mse'])

model3 = Sequential()
model3.add(Dense(input_shape=(180,), units=80, activation=sigmoid))
model3.add(Dense(80, activation=sigmoid))
model3.add(Dense(80, activation=sigmoid))
model3.add(Dense(1, activation=linear))
model3.compile(optimizer="adam", loss=mean_squared_error, metrics=['mse'])

mses_model1 = cross_validate(model, x_train, y_train, epochs=20, verbose=0)
mses_model2 = cross_validate(model2, x_train, y_train, epochs=20, verbose=0)
mses_model3 = cross_validate(model3, x_train, y_train, epochs=20, verbose=0)

# RNN
featu_enc, x, y = get_data(data, dynamic=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

rnn_model = Sequential()
rnn_model.add(Embedding(input_dim=21, output_dim=50, input_length=18))
rnn_model.add(LSTM(100, return_sequences=False))
rnn_model.add(Dense(1, activation=linear))
rnn_model.compile(optimizer="adam", loss=mean_squared_error, metrics=['mse'])

rnn_model2 = Sequential()
rnn_model2.add(Embedding(input_dim=21, output_dim=50, input_length=18))
rnn_model2.add(LSTM(100, return_sequences=True))
rnn_model2.add(LSTM(100, return_sequences=False))
rnn_model2.add(Dense(1, activation=linear))
rnn_model2.compile(optimizer="adam", loss=mean_squared_error, metrics=['mse'])

rnn_model3 = Sequential()
rnn_model3.add(Embedding(input_dim=21, output_dim=100, input_length=18))
rnn_model3.add(LSTM(100, return_sequences=True))
rnn_model3.add(LSTM(100, return_sequences=False))
rnn_model3.add(Dense(1, activation=linear))
rnn_model3.compile(optimizer="adam", loss=mean_squared_error, metrics=['mse'])

mses_rnn1 = cross_validate(rnn_model, x_train, y_train, epochs=20, verbose=0)
mses_rnn2 = cross_validate(rnn_model2, x_train, y_train, epochs=20, verbose=0)
mses_rnn3 = cross_validate(rnn_model3, x_train, y_train, epochs=20, verbose=0)

print(np.mean(mses_model1), np.mean(mses_model2), np.mean(mses_model3))
print(np.mean(mses_rnn1), np.mean(mses_rnn2), np.mean(mses_rnn3))
