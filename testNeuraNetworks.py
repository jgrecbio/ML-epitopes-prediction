from keras import Sequential
import numpy as np
import unittest
from keras.layers import Conv1D, MaxPooling1D, Reshape
from keras.activations import sigmoid, linear, relu
from keras.optimizers import sgd, rmsprop, adadelta, adagrad, adam, adamax, nadam, SGD
from keras.losses import mean_squared_error
from neural_net import dense_model, get_optimizer_params
from cnn import cnn_model, conv_operation
from rnn import rnn_model


class TestNeuralNetworks(unittest.TestCase):
    def testCompileParams(self):
        lr, m1, m2, rho, epsilon, decay = 0.01, 0.9, 0.999, 0.95, 1e-8, 0.
        self.assertEqual(get_optimizer_params(sgd, lr, m1, m2, epsilon, decay, rho),
                         {"lr": lr})
        self.assertEqual(get_optimizer_params(SGD, lr, m1, m2, epsilon, decay, rho),
                         {"lr": lr})
        self.assertEqual(get_optimizer_params(adam, lr, m1, m2, epsilon, decay, rho),
                         {"lr": lr, "beta_1": m1, "beta_2": m2,
                          "epsilon": epsilon, "decay": decay})

    def testDenseModel(self):
        data, labels = np.random.random((100, 180)), np.random.random(100)
        nb_units_false = [40, 1]
        nb_units_true = [40, 40, 1]
        activations = [sigmoid, sigmoid, linear]
        input_shape = (180,)
        self.assertRaises(ValueError, dense_model, *[nb_units_false,
                                                     activations,
                                                     input_shape])
        lr, m1, m2, rho, epsilon, decay = 0.01, 0.9, 0.999, 0.95, 1e-8, 0.
        for opt in (sgd, rmsprop, adamax, adamax, adam, adagrad, adadelta, nadam):
            model = dense_model(nb_units=nb_units_true,
                                activations=activations,
                                input_shape=input_shape,
                                optimizer=opt,
                                loss=mean_squared_error,
                                learning_rate=lr, momentum_1=m1,
                                momentum_2=m2, epsilon=epsilon,
                                decay=decay, rho=rho)
            model.fit(x=data, y=labels, epochs=1)

    def testPreNet(self):
        data, labels = np.random.random((100, 180)), np.random.random(100)
        pre_net = dense_model(nb_units=40, activations=sigmoid, input_shape=(180,))
        post_net = dense_model(nb_units=[40, 1], activations=[sigmoid, linear], pre_model=pre_net)
        post_net.fit(x=data, y=labels, epochs=1)

    def testCNN(self):
        data, labels = np.random.random((100, 180, 1)), np.random.random(100)
        conv_layout = [
            conv_operation(Conv1D, 64, 2, 1, "valid", "relu", MaxPooling1D, 2),
            conv_operation(Conv1D, 32, 2, 1, "valid", "relu", MaxPooling1D, 2)
        ]
        model = cnn_model(conv_layout, input_shape=(180, 1), dense_nb_neurons=[40, 40, 1],
                          dense_activations=[sigmoid, sigmoid, linear])
        model.fit(data, labels, epochs=1)

    def testReshapePrenet(self):
        data, labels = np.random.random((100, 180)), np.random.random(100)
        conv_layout = [
            conv_operation(Conv1D, 64, 2, 1, "valid", "relu", MaxPooling1D, 2),
            conv_operation(Conv1D, 32, 2, 1, "valid", "relu", MaxPooling1D, 2)
        ]

        model = Sequential()
        model.add(Reshape((180, 1), input_shape=(180,)))

        model = cnn_model(conv_layout, dense_nb_neurons=[40, 40, 1],
                          dense_activations=[sigmoid, sigmoid, linear],
                          pre_model=model)
        model.fit(data, labels, epochs=1)

    def testRNN(self):
        data, labels = np.random.randint(size=(100, 9), low=1, high=20), np.random.random(100)
        model = rnn_model(dense_nb_neurons=[40, 1], dense_activations=[relu, linear])
        model.summary()
        model.fit(data, labels, epochs=1)

    def testReg(self):
        data, labels = np.random.random((100, 180)), np.random.random(100)
        nb_units_false = [40, 1]
        nb_units_true = [40, 40, 1]
        activations = [sigmoid, sigmoid, linear]
        input_shape = (180,)
        self.assertRaises(ValueError, dense_model, *[nb_units_false,
                                                     activations,
                                                     input_shape])
        model = dense_model(nb_units=nb_units_true,
                            activations=activations,
                            input_shape=input_shape,
                            l1_regularization=[0.01, 0.01, 0.01],
                            dropout_reg=0.5)
        model.fit(x=data, y=labels, epochs=1)
