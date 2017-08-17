#! /usr/bin/env python
from argparse import ArgumentParser

import numpy as np
from settings import input_size, use_PCA, load_folder_name
from sklearn.externals import joblib

from models import NeuralNetwork


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input', type=float, nargs=input_size)
    return parser.parse_args()


def get_features_evaluate(arguments):
    features = np.array(arguments.input, dtype=np.float32).reshape((1, input_size))
    evaluate(features)


def evaluate(features):
    nn = NeuralNetwork(layers=(15, 16, 8), learning_rate=0.001, path=load_folder_name + "/model.ckpt")

    if use_PCA:
        input_normalizer = joblib.load(load_folder_name + "/data_normalizer.pkl")
        input_PCA = joblib.load(load_folder_name + "/data_PCA.pkl")
        features = input_normalizer.transform(features)
        features = input_PCA.transform(features)

    input_scaler = joblib.load(load_folder_name + "/data_scaler.pkl")
    output_scaler = joblib.load(load_folder_name + "/label_scaler.pkl")
    features = input_scaler.transform(features)

    output = nn.feed_forward(features)[0, :]

    output = output_scaler.inverse_transform(output)
    print(output)


if __name__ == "__main__":
    args = parse_args()
    get_features_evaluate(args)
