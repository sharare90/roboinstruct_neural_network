# data_directory = "/home/sharare/SelfStudies/Robotics/RecurrentNeuralNetworkforRoboinstruct/ScriptData"
data_directory = "/home/siavash/programming/RoboInstructDropboxUploader/data/trajectories/script/numpy/train"
# valid_data_directory = "/home/sharare/SelfStudies/Robotics/RecurrentNeuralNetworkforRoboinstruct/NewData/numpy"

input_size_PCA = 15
input_size = 15
first_hidden_layer = 16
use_PCA = False
load_folder_name = "./states/last"
save_folder_name = "./states/last"


def hidden_layer_input(x, weights, bias):
    import tensorflow as tf
    y1 = 1.7159 * (tf.tanh(tf.matmul(2 * x / 3, weights) + bias))
    # y1 = tf.nn.relu(tf.matmul(x, weights) + bias)
    return y1
