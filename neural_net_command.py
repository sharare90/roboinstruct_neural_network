#! /usr/bin/env python3
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from settings import input_size_PCA, input_size, use_PCA, first_hidden_layer, load_folder_name
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

    W1 = tf.Variable(
        tf.random_normal(shape=[input_size_PCA, first_hidden_layer], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
    b1 = tf.Variable(tf.random_normal(shape=[first_hidden_layer], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
    # y1 = 1.7159 * (tf.tanh(tf.matmul(2 * x / 3, W1) + b1))
    y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.random_normal(shape=[first_hidden_layer, 8], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
    b2 = tf.Variable(tf.random_normal(shape=[8], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))

    y = tf.matmul(y1, W2) + b2
    saver = tf.train.Saver()
    if use_PCA:
        input_normalizer = joblib.load(load_folder_name + "/data_normalizer.pkl")
        input_PCA = joblib.load(load_folder_name + "/data_PCA.pkl")
        features = input_normalizer.transform(features)
        features = input_PCA.transform(features)

    input_scaler = joblib.load(load_folder_name + "/data_scaler.pkl")
    output_scaler = joblib.load(load_folder_name + "/label_scaler.pkl")
    features = input_scaler.transform(features)
    with tf.Session() as sess:
        saver.restore(sess, load_folder_name + "/model.ckpt")
        output = sess.run(y, feed_dict={x: features})[0, :]
    output = output_scaler.inverse_transform(output)
    print(output)


if __name__ == "__main__":
    args = parse_args()
    get_features_evaluate(args)
