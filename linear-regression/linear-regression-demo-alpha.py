import tensorflow as tf
import random
from matplotlib import pyplot as plt

DISABLE_GPU = True
if DISABLE_GPU:
    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        pass

#
# Generate data set
#

num_features = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
X_inputs = tf.random.normal((num_examples, num_features), stddev = 1)
y_inputs = true_w[0] * X_inputs[:, 0] + true_w[1] * X_inputs[:, 1] + true_b
y_inputs += tf.random.normal(y_inputs.shape, stddev = 0.01)

print(X_inputs[0])
print(y_inputs[0])

#
# Plot
#

def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(X_inputs[:, 1], y_inputs, 1)
# plt.show()

#
# Read batch data
#

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = indices[i : min(i + batch_size, num_examples)]
        yield tf.gather(features, axis = 0, indices = j), tf.gather(labels, axis = 0, indices = j)

batch_size = 10

for X, y in data_iter(batch_size, X_inputs, y_inputs):
    print(X, y)
    break

#
# Prepare parameters
# 

w = tf.Variable(tf.random.normal((num_features, 1), stddev = 0.01))
b = tf.Variable(tf.zeros((1,)))

def linreg(X, w, b):
    return tf.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2

def sgd(params, learning_rate, batch_size, grads):
    for i, param in enumerate(params):
        param.assign_sub(learning_rate * grads[i] / batch_size)

#
# Train
#

learning_rate = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, X_inputs, y_inputs):
        with tf.GradientTape() as t:
            t.watch([w, b])
            l = tf.reduce_sum(loss(net(X, w, b), y))
            grads = t.gradient(l, [w, b])
            sgd([w, b], learning_rate, batch_size, grads)
    train_l = loss(net(X_inputs, w, b), y_inputs)
    print('epoch %d, loss %f' % (epoch + 1, tf.reduce_mean(train_l)))

print(true_w, w.numpy().tolist())
print(true_b, b.numpy().tolist())
