# Notes

## TensorFlow Notes
### tf.gather
#### Simple demo
```
import tensorflow as tf

x = tf.constant([1, 2, 3, 4, 5])

y = tf.gather(x, [0, 1])
print(y)
```

#### Demo with axis
```
import tensorflow as tf

x = tf.constant([[1, 2, 3],[4, 5, 6],[7, 8, 9]])

y = tf.gather(x, axis = 0, indices = [0, 1])
print(y)
# [[1 2 3], 
#  [4 5 6]]

z = tf.gather(x, axis = 1, indices = [0, 1])
print(z)
# [[1 2]
#  [4 5]
#  [7 8]]
```

## Python Notes
### Yield
```
import random

def number_generator(n):
    nums = list(range(n))
    random.shuffle(nums)
    for num in nums:
        yield num

gen = number_generator(5)

for num in gen:
    print(num)
```
