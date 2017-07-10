import numpy as np


features = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.float32)


def evaluate(features):
    nn = load_neural_network()

    print()