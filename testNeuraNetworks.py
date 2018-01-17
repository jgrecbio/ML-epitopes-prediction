from keras import Sequential
import numpy as np
import unittest
from keras.layers import Conv1D, MaxPooling1D, Reshape
from keras.activations import sigmoid, linear, relu
from keras.optimizers import SGD
from keras.losses import mean_squared_error
from neural_net import dense_model, compile_model
from cnn import cnn_model, conv_operation
from rnn import rnn_model


class TestNeuralNetworks(unittest.TestCase):
    def testDenseModel(self):
        data, labels = np.random.random((100, 180)), np.random.random(100)
        nb_units_false = [40, 1]
        nb_units_true = [40, 40, 1]
        activations = [sigmoid, sigmoid, linear]
        input_shape = (180,)
        self.assertRaises(ValueError, dense_model, *[nb_units_false,
                                                     activations,
                                                     input_shape])
        uncompiled_model = dense_model(nb_units=nb_units_true,
                                       activations=activations,
                                       input_shape=input_shape)
        model = compile_model(model=uncompiled_model, optimizer=SGD,
                              loss=mean_squared_error, **{"lr": 0.01})
        model.fit(x=data, y=labels, epochs=1)

    def testPreNet(self):
        data, labels = np.random.random((100, 180)), np.random.random(100)
        pre_net = dense_model(nb_units=40, activations=sigmoid, input_shape=(180, ))
        post_net = dense_model(nb_units=[40, 1], activations=[sigmoid, linear], pre_model=pre_net)
        post_net_compiled = compile_model(post_net, SGD, mean_squared_error, *[0.01])
        post_net_compiled.fit(x=data, y=labels, epochs=1)

    def testCNN(self):
        data, labels = np.random.random((100, 180, 1)), np.random.random(100)
        conv_layout = [
            conv_operation(Conv1D, 64, 2, 1, "valid", "relu", MaxPooling1D, 2),
            conv_operation(Conv1D, 32, 2, 1, "valid", "relu", MaxPooling1D, 2)
        ]
        model = cnn_model(conv_layout, input_shape=(180, 1), dense_nb_neurons=[40, 40, 1],
                          dense_activations=[sigmoid, sigmoid, linear])
        model = compile_model(model, SGD, mean_squared_error, **{"lr": 0.01})
        model.fit(data, labels, epochs=1)

    def testReshapePrenet(self):
        data, labels = np.random.random((100, 180)), np.random.random(100)
        conv_layout = [
            conv_operation(Conv1D, 64, 2, 1, "valid", "relu", MaxPooling1D, 2),
            conv_operation(Conv1D, 32, 2, 1, "valid", "relu", MaxPooling1D, 2)
        ]

        model = Sequential()
        model.add(Reshape((180, 1), input_shape=(180, )))

        model = cnn_model(conv_layout, dense_nb_neurons=[40, 40, 1],
                          dense_activations=[sigmoid, sigmoid, linear],
                          pre_model=model)
        compile_model(model, SGD, mean_squared_error, *[0.01])
        model.fit(data, labels, epochs=1)

    def testRNN(self):
        data, labels = np.random.randint(size=(100, 9), low=1, high=20), np.random.random(100)
        model = compile_model(rnn_model(dense_nb_neurons=[40, 1],
                                        dense_activations=[relu, linear]), SGD, mean_squared_error,
                              *[0.01])
        model.summary()
        model.fit(data, labels, epochs=1)
