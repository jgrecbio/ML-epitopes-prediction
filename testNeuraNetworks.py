import numpy as np
import unittest
from keras.activations import sigmoid, linear
from keras.optimizers import SGD
from keras.losses import mean_squared_error
from neural_net import dense_model, compile_model


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
        post_net_compiled.fit(x=data, y=labels)
