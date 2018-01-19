from typing import List, Optional
from keras import Sequential
from keras.layers import GRU, Embedding, Bidirectional, Dropout, Flatten
from keras_implementation.neural_net import dense_model


def rnn_model(input_dim: int=20,
              output_dim: int=50,
              input_length: int=9,
              nb_rnn_neurons: List[int]=None,
              dense_nb_neurons: Optional[List[int]]=None,
              dense_activations: Optional[List]=None,
              rnn_cell_type=GRU, pre_model=None, dropout: int=0.5,
              name: Optional[str]=None,
              *args, **kwargs) -> Sequential:
    """
    Stacked RNN model
    :param input_dim: length of the vocabulary of vectors
    :param output_dim: nb of components of aa vectors
    :param input_length: length of tokenized vectors
    :param nb_rnn_neurons: nb of neurons per RNN layers
    :param dense_nb_neurons: nb of neurons per dense layer
    :param dense_activations: type of activation per dense layer
    :param rnn_cell_type: type RNN
    :param pre_model: keras Sequential
    :param dropout: rate of dropout for LSTM regularization
    :param name: name of the neural network
    :param args: extra param to dense model
    :param kwargs: extra param to dense model
    :return:
    """

    model = Sequential(name=name) if not pre_model else pre_model

    # Embedding layer
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))

    # Stacked RNN
    rnn_neurons = nb_rnn_neurons if nb_rnn_neurons else [32]
    for nb_neurons in rnn_neurons:
        model.add(Bidirectional(rnn_cell_type(nb_neurons, return_sequences=True)))
        model.add(Dropout(dropout))

    # Dens model
    model.add(Flatten())
    if dense_activations and dense_nb_neurons:
        return dense_model(nb_units=dense_nb_neurons, activations=dense_activations, pre_model=model,
                           *args, **kwargs)

    return model
