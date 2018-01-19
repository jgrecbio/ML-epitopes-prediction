import numpy as np
from functools import reduce
import tensorflow as tf


x = tf.placeholder(tf.float32, [None, 180])
y = tf.placeholder(tf.float32, [None, 1])

ws = [tf.Variable(tf.random_normal((180, 40), dtype=tf.float32)),
      tf.Variable(tf.random_normal((40, 20)), dtype=tf.float32),
      tf.Variable(tf.random_normal((20, 1), dtype=tf.float32))]

bs = [tf.Variable(tf.zeros([40])),
      tf.Variable(tf.zeros([20])),
      tf.Variable(tf.zeros([1]))]

activations = [
      tf.nn.relu, tf.nn.relu, tf.identity
]


def reduce_dynamic(func):
    def dynamic_dispatch(x, y):
        func(x, *y)
    return dynamic_dispatch


@reduce_dynamic
def layer_operation(x, w, b, activation):
    print(x)
    i = activation(tf.add(tf.matmul(x, w), b))
    print("af", i)
    return i


def forward_reduce(x: tf.Tensor, ws, bs, activations):
    return reduce(layer_operation, zip(ws, bs, activations), x)


init = tf.global_variables_initializer()
# y_pred = forward_reduce(x, ws, bs, activations)
# loss = tf.square(y_pred - y)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
with tf.Session() as sess:
    sess.run(init)
    y_pred = sess.run(forward_reduce(x, ws, bs, activations), feed_dict={x: np.random.random((100, 180))})
