from keras import Sequential
from keras.layers import GRU, Embedding, Dense, Bidirectional, Dropout, Flatten
from keras.activations import linear, relu


def rnn_model():
    model = Sequential()
    model.add(Embedding(input_dim=20, output_dim=50, input_length=9))
    model.add(GRU(30, return_sequences=True))
    model.add(Bidirectional(GRU(32, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(40, activation=relu))
    model.add(Dense(1, activation=linear))
    return model
