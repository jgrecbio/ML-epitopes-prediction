from typing import List, Optional
from keras import Sequential
from keras.layers import GRU, Embedding, Bidirectional, Dropout, Flatten
from neural_net import dense_model


def rnn_model(input_dim: int=20,
              output_dim: int=50,
              input_length: int=9,
              dense_nb_neurons: Optional[List[int]]=None,
              dense_activations: Optional[List]=None,
              rnn_cell_type=GRU, pre_model=None, dropout: int=0.5) -> Sequential:

    model = Sequential() if not pre_model else pre_model

    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    model.add(Bidirectional(rnn_cell_type(32, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Flatten())

    if dense_activations and dense_nb_neurons:
        return dense_model(nb_units=dense_nb_neurons, activations=dense_activations, pre_model=model)

    return model
