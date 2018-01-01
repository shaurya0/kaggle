import numpy as np
from ThreadSafeGenerator import ThreadSafeGenerator
import librosa
from os.path import join as path_join
from tools import preprocess_recording
import glob


class WavDataGenerator(object):
    def __init__(self, data_dir, labels, nx, ny, is_train=False, batch_size=32, sampling_rate=16000, shuffle=True):
        self.labels_str = labels
        self.num_labels = len(labels)
        self.files = list()
        self.labels = list()
        self.shuffle = shuffle
        self.sampling_rate = sampling_rate
        self.nx = nx
        self.ny = ny

        total_num_files = 0

        if is_train:
            self.subfolders = [path_join(data_dir, l) for l in labels]
            for (i, sf) in enumerate(self.subfolders):
                files = glob.glob(path_join(sf, '*.wav'))
                num_files = len(files)
                total_num_files += num_files

                self.files.extend(files)
                self.labels.extend([i] * num_files)
        else:
            files = glob.glob(path_join(data_dir, '*.wav'))
            num_files = len(files)
            total_num_files += num_files
            self.files.extend(files)

        self.is_train = is_train
        self.files = np.array(self.files)
        self.labels = np.array(self.labels, dtype=np.int32)
        self.batch_size = batch_size
        self._num_examples = total_num_files
        self.indices = np.arange(0, total_num_files, dtype=np.int32)
        self.idx = 0

    @property
    def num_examples(self):
        return self._num_examples


    def _shuffle_data(self):
        np.random.shuffle(self.indices)
        self.files = self.files[self.indices]
        self.labels = self.labels[self.indices]

    def _load_batch(self, indices, batch_size):
        batch_data = np.empty((batch_size, self.nx, self.ny, 1), dtype=np.float32)
        batch_labels = np.zeros((batch_size, self.num_labels), dtype=np.float32)
        for (i, k) in enumerate(indices):
            file_path = self.files[k]
            raw_audio, _ = librosa.load(file_path, sr=self.sampling_rate)
            batch_data[i] = preprocess_recording(raw_audio, sr=self.sampling_rate)

            if self.is_train:
                label = self.labels[k]
                batch_labels[i][label] = 1.0

        return batch_data, batch_labels

    @ThreadSafeGenerator
    def generator(self):
        assert(self.is_train == True)

        batch_size = self.batch_size
        indices = np.zeros((batch_size,), dtype=np.int32)
        start = 0
        end = self._num_examples - batch_size

        while True:
            if self.shuffle:
                self._shuffle_data()

            for i in range(start, end, batch_size):
                indices = np.arange(i, i + batch_size)
                batch_data, batch_labels = self._load_batch(indices, batch_size)
                yield batch_data, batch_labels

    @ThreadSafeGenerator
    def single_pass_generator(self):
        assert(self.is_train == False)
        finished = False
        batch_size = self.batch_size
        indices = np.zeros((batch_size,), dtype=np.int32)
        start = 0
        end = self._num_examples

        for i in range(start, end, batch_size):
            if i + batch_size >= end:
                break
            indices = np.arange(i, i + batch_size)
            batch_data, _ = self._load_batch(indices, batch_size)
            yield batch_data

        if (end%batch_size) == 0:
            finished = True
            return
        else:
            if finished:
                return

            indices = np.arange(i, end)
            print(indices)
            batch_size = len(indices)
            finished = True
            batch_data, _ = self._load_batch(indices, batch_size)
            yield batch_data






