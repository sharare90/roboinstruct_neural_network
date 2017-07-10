from os import listdir
from os.path import join

import numpy as np

from settings import data_directory


class Database(object):
    def __init__(self, data_directory):
        self._input_matrices = []
        self._label_matrices = []
        self.data = None
        self.labels = None
        self.test_data = None
        self.test_labels = None
        self._data_directory = data_directory
        self._create_data()
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = 100

    def _create_data(self):
        for file_name in listdir(self._data_directory):
            position_data = np.load(join(self._data_directory, file_name))
            self._input_matrices.append(position_data[:-1, :])
            self._label_matrices.append(position_data[1:, :8])

        self.data = np.concatenate(self._input_matrices[:100])
        self.labels = np.concatenate(self._label_matrices[:100])

        self.test_data = np.concatenate(self._input_matrices[100:])
        self.test_labels = np.concatenate(self._label_matrices[100:])

    def get_length(self):
        return len(self.data)

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self.data = self.data[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.data[start:end], self.labels[start:end]

db = Database(data_directory)


if __name__ == "__main__":
    print(db.get_length())
