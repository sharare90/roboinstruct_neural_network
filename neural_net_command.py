import numpy as np
import tensorflow as tf


features = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.float32)


def evaluate(features):
    x = tf.placeholder(tf.float32, [None, 15])
    W = tf.Variable(tf.random_normal(shape=[15, 8], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
    b = tf.Variable(tf.random_normal(shape=[8], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))

    y = tf.matmul(x, W) + b
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "/home/sharare/PycharmProjects/roboinstruct_training/model.ckpt")
        output = sess.run(y, feed_dict={x: features})

    print(output)

evaluate(features)