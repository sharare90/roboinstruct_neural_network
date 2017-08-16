import tensorflow as tf
from database import db

# from settings import input_size_PCA
from models import NeuralNetwork


batch_xs = db.data
batch_ys = db.labels
save_path = "states/last2/model.ckpt"

nn = NeuralNetwork(layers=(15, 10, 8))
nn.train(batch_xs, batch_ys, iterations=1000)

print(nn.evaluate_cost(db.test_data, db.test_labels))
