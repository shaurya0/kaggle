import numpy as np
import librosa
from os.path import join as path_join
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

    def _load_sample(self, idx):
        X = np.empty((self.nx, self.ny), dtype=np.float32)
        y = np.zeros((self.num_labels), dtype=np.float32)
        # for (i, k) in enumerate(indices):
        file_path = self.files[idx]
        label = self.labels[idx]
        raw_audio, _ = librosa.load(file_path, sr=self.sampling_rate)
        X = self._preprocess_recording(raw_audio)
        y[label] = 1.0

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
