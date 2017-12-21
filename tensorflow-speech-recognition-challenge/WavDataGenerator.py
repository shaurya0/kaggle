import numpy as np
from ThreadSafeGenerator import ThreadSafeGenerator
import librosa
from os.path import join as path_join
import glob


class WavDataGenerator(object):
    def __init__(self, data_dir, labels, nx, ny, batch_size=32, sampling_rate=16000, shuffle=True):
        self.labels_str = labels
        self.num_labels = len(labels)
        self.subfolders = [path_join(data_dir, l) for l in labels]
        self.files = list()
        self.labels = list()
        self.shuffle = shuffle
        self.sampling_rate = sampling_rate
        self.nx = nx
        self.ny = ny

        total_num_files = 0
        for (i, sf) in enumerate(self.subfolders):
            files = glob.glob(path_join(sf, '*.wav'))
            num_files = len(files)
            total_num_files += num_files

            self.files.extend(files)
            self.labels.extend([i] * num_files)

        self.files = np.array(self.files)
        self.labels = np.array(self.labels, dtype=np.int32)
        self.batch_size = batch_size
        self._num_examples = total_num_files
        self.indices = np.arange(0, total_num_files, dtype=np.int32)
        self.idx = 0

    @property
    def num_examples(self):
        return self._num_examples

    def _preprocess_recording(self, x):
        length = len(x)
        sr = self.sampling_rate
        if length < sr:
            zero_pad_length = sr - length
            tmp = np.zeros((zero_pad_length, ), dtype=np.float32)
            x = np.concatenate((tmp, x), axis=0)

        return librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128, fmax=8000, hop_length=512)

    def _shuffle_data(self):
        np.random.shuffle(self.indices)
        self.files = self.files[self.indices]
        self.labels = self.labels[self.indices]

    def _load_batch(self, indices, batch_data, batch_labels):
        for (i, k) in enumerate(indices):
            file_path = self.files[k]
            label = self.labels[k]
            X, _ = librosa.load(file_path, sr=self.sampling_rate)
            batch_data[i] = self._preprocess_recording(X)
            batch_labels[i][label] = 1.0

        return batch_data, batch_labels

    @ThreadSafeGenerator
    def generator(self):
        batch_size = self.batch_size
        indices = np.zeros((batch_size,), dtype=np.int32)
        start = 0
        end = self._num_examples - batch_size
        batch_data = np.empty((batch_size, self.nx, self.ny), dtype=np.float32)
        batch_labels = np.zeros((batch_size, self.num_labels), dtype=np.float32)
        while True:
            if self.shuffle:
                self._shuffle_data()
            for i in range(start, end, batch_size):
                indices = np.arange(i, i + batch_size)
                batch_data, batch_labels = self._load_batch(indices, batch_data, batch_labels)
                yield batch_data, batch_labels
