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
y_inputs = true_w[0] * X_inputs[:,0] + true_w[1] * X_inputs[:,1] + true_b
y_inputs += tf.random.normal(y_inputs.shape, stddev = 0.01)


#
# Read batch data
#

from tensorflow import data as tfdata

batch_size = 10
dataset = tfdata.Dataset.from_tensor_slices((X_inputs, y_inputs))
dataset = dataset.shuffle(buffer_size = num_examples) 
dataset = dataset.batch(batch_size)
data_iter = iter(dataset)

for X, y in data_iter:
    print(X, y)
    break

for (batch, (X, y)) in enumerate(dataset):
    print(X, y)
    break

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import initializers as init

model = keras.Sequential()
model.add(layers.Dense(1, kernel_initializer = init.RandomNormal(stddev = 0.01)))

from tensorflow import losses
loss = losses.MeanSquaredError()

from tensorflow.keras import optimizers
trainer = optimizers.SGD(learning_rate = 0.03)

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for (batch, (X, y)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            l = loss(model(X, training = True), y)

            grads = tape.gradient(l, model.trainable_variables)
            trainer.apply_gradients(zip(grads, model.trainable_variables))
            
    l = loss(model(X_inputs), y_inputs)
    print('epoch %d, loss: %f' % (epoch, l))

print(true_w, model.get_weights()[0])
print(true_b, model.get_weights()[1])
