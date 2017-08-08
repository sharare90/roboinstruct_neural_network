from os import listdir
from os.path import join

import numpy as np
from sklearn import preprocessing, decomposition
from sklearn.externals import joblib

from settings import data_directory, input_size_PCA, use_PCA


class Database(object):
    def __init__(self, data_directory):
        self._input_matrices = []
        self._label_matrices = []
        self.data = None
        self.labels = None
        self.test_data = None
        self.test_labels = None
        self._data_directory = data_directory
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.num_examples = 1
        self.data_scaler = preprocessing.StandardScaler()
        self.label_scaler = preprocessing.StandardScaler()
        if use_PCA:
            self.data_normalizer = preprocessing.StandardScaler(with_std=False)
            self.data_PCA = decomposition.PCA(n_components=input_size_PCA)
        self._create_data()

    def _create_data(self):
        for file_name in listdir(self._data_directory):
            position_data = np.load(join(self._data_directory, file_name))
            self._input_matrices.append(position_data[:-1, :])
            self._label_matrices.append(position_data[1:, :8])

        if self.num_examples != 1:
            self.data = np.concatenate(self._input_matrices[:self.num_examples])
            self.labels = np.concatenate(self._label_matrices[:self.num_examples])
            self.test_data = np.concatenate(self._input_matrices[self.num_examples:])
            self.test_labels = np.concatenate(self._label_matrices[self.num_examples:])
        else:
            self.data = self._input_matrices[0]
            self.labels = self._label_matrices[0]
            self.test_data = self._input_matrices[0]
            self.test_labels = self._label_matrices[0]
        self.data_preprocess()

        self.label_scaler.fit(self.labels)
        joblib.dump(self.label_scaler, './states/last/label_scaler.pkl')
        self.labels = self.label_scaler.transform(self.labels)

        self.test_labels = self.label_scaler.transform(self.test_labels)

    def get_length(self):
        return len(self.data)

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.data = self.data[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self._index_in_epoch
        return self.data[start:end], self.labels[start:end]

    def data_preprocess(self):
        if use_PCA:
            self.data_normalizer.fit(self.data)
            joblib.dump(self.data_normalizer, './states/last/data_normalizer.pkl')
            self.data = self.data_normalizer.transform(self.data)
            self.test_data = self.data_normalizer.transform(self.test_data)
            self.data_PCA.fit(self.data)
            joblib.dump(self.data_PCA, './states/last/data_PCA.pkl')
            self.data = self.data_PCA.transform(self.data)
            self.test_data = self.data_PCA.transform(self.test_data)
            print(self.data_PCA.components_)
            print(self.data_PCA.explained_variance_)

        self.data_scaler.fit(self.data)
        joblib.dump(self.data_scaler, './states/last/data_scaler.pkl')
        self.data = self.data_scaler.transform(self.data)
        self.test_data = self.data_scaler.transform(self.test_data)


db = Database(data_directory)

if __name__ == "__main__":
    print(db.get_length())
