#! /usr/bin/env python3
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from sklearn.externals import joblib


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input', type=float, nargs=15)
    return parser.parse_args()


def get_features_evaluate(arguments):
    features = np.array(arguments.input, dtype=np.float32).reshape((1, 15))
    evaluate(features)


def evaluate(features):
    x = tf.placeholder(tf.float32, [None, 15])
    W = tf.Variable(tf.random_normal(shape=[15, 8], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
    b = tf.Variable(tf.random_normal(shape=[8], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))

    y = tf.matmul(x, W) + b
    saver = tf.train.Saver()
    input_normalizer = joblib.load("/states/last/data_normalizer.pkl")
    input_PCA = joblib.load("/states/last/data_PCA.pkl")
    input_scaler = joblib.load("/states/last/data_scaler.pkl")
    output_scaler = joblib.load("/states/last/label_scaler.pkl")
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
