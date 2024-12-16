import tensorflow as tf
import math
import time

rotate_angle = 5.
radian_min = rotate_angle * math.pi / 180.
radian_max = rotate_angle * math.pi / 180.
random_seed = int(round(time.time()))
print('=============== MAX =======================')
print(radian_max)
print('=============== MIN =======================')
print(radian_min)
print('================ RANDOM =====================')
random_angle = tf.random_uniform(shape=[1], minval=radian_min, maxval=radian_max, seed=random_seed)

with tf.Session() as sess:
    result = sess.run(random_angle)
    print(result)
