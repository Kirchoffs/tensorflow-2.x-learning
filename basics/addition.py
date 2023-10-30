import tensorflow as tf
from time import time

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

a = tf.ones((1000,))
b = tf.ones((1000,))

# Method 1:
start = time()
c1 = tf.Variable(tf.zeros((1000,)))
for i in range(1000):
    c1[i].assign(a[i] + b[i])
print("Method 1: ", time() - start)

# Method 2:
start = time()
c2 = tf.Variable(tf.zeros((1000,)))
c2.assign(a + b)
print("Method 2: ", time() - start)
