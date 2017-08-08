#! /usr/bin/env python3
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from settings import input_size_PCA, input_size
from sklearn.externals import joblib


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input', type=float, nargs=input_size)
    return parser.parse_args()


def get_features_evaluate(arguments):
    features = np.array(arguments.input, dtype=np.float32).reshape((1, input_size))
    evaluate(features)


def evaluate(features):
    x = tf.placeholder(tf.float32, [None, input_size_PCA])

    W1 = tf.Variable(tf.random_normal(shape=[input_size_PCA, 16], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
    b1 = tf.Variable(tf.random_normal(shape=[16], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
    y1 = tf.matmul(x, W1) + b1

    W2 = tf.Variable(tf.random_normal(shape=[16, 8], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
    b2 = tf.Variable(tf.random_normal(shape=[8], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))

    y = tf.matmul(y1, W2) + b2
    saver = tf.train.Saver()
    input_normalizer = joblib.load("./states/last/data_normalizer.pkl")
    input_PCA = joblib.load("./states/last/data_PCA.pkl")
    input_scaler = joblib.load("./states/last/data_scaler.pkl")
    output_scaler = joblib.load("./states/last/label_scaler.pkl")
    features = input_normalizer.transform(features)
    features = input_PCA.transform(features)
    features = input_scaler.transform(features)
    with tf.Session() as sess:
        saver.restore(sess, "./states/last/model.ckpt")
        output = sess.run(y, feed_dict={x: features})[0, :]
    output = output_scaler.inverse_transform(output)
    print(output)


if __name__ == "__main__":
    args = parse_args()
    get_features_evaluate(args)
