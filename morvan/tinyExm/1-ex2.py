import tensorflow as tf
import numpy as np

# input data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# model
weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.zeros([1]))
y = weight * x_data + bias

# loss
loss = tf.reduce_mean(tf.square(y - y_data))

# GD
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 0.5 is stepLength?
train = optimizer.minimize(loss)

# init
init = tf.global_variables_initializer()

# session.use to run and print 
sess = tf.Session()
sess.run(init)

# iter
for step in range(201):
    sess.run(train)
    if (step % 20 == 0):
        print(step, sess.run(weight), sess.run(bias))
