import numpy as np
import librosa
from os.path import join as path_join
import glob


class WavDataLoader(object):
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

        self.load_data()

    def load_data(self):
        start = 0
        end = self._num_examples
        if self.shuffle:
            self._shuffle_data()

        X = np.empty((self._num_examples, self.nx, self.ny), dtype=np.float32)
        y = np.zeros((self._num_examples), dtype=np.float32)


        for i in range(start, end):
            X[i], y[i] = self._load_sample(i)             

        self.X = X
        self.y = y


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
        file_path = self.files[idx]
        label = self.labels[idx]
        raw_audio, _ = librosa.load(file_path, sr=self.sampling_rate)
        X = self._preprocess_recording(raw_audio)

        return X, label


if __name__ == "__main__":
    labels = ['silence', 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    data_dir = r'C:\Development\kaggle\tensorflow-speech-recognition-challenge\data\train\audio'
    wdl = WavDataLoader(data_dir, labels, 128,32)
    print('hello world')