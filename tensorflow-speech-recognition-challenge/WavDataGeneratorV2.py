import numpy as np
from ThreadSafeGenerator import ThreadSafeGenerator
import librosa
from os.path import join as path_join
from tools import preprocess_recording
import glob



class WavDataGeneratorV2(object):
    def __init__(self, data_dir, labels, nx, ny, sampling_rate=16000, shuffle=True):
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

    def _load_sample(self, idx):
        X = np.empty((self.nx, self.ny,1), dtype=np.float32)
        y = np.zeros((self.num_labels), dtype=np.float32)
        label = self.labels[idx]
        y[label] = 1.0

        file_path = self.files[idx]
        raw_audio, _ = librosa.load(file_path, sr=self.sampling_rate)
        X = preprocess_recording(raw_audio, sr=self.sampling_rate)

        return X, y

    def generator(self):
        start = 0
        end = self._num_examples

        while True:
            if self.shuffle:
                self._shuffle_data()
            for i in range(start, end):
                X, y = self._load_sample(i)
                yield X, y
