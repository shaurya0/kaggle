import numpy as np
import librosa
from os.path import join as path_join
from tools import preprocess_recording_dict
import glob


class WavDataLoader(object):
    def __init__(self, data_dir, labels, sampling_rate=16000, shuffle=True):
        self.labels_str = labels
        self.num_labels = len(labels)
        self.subfolders = [path_join(data_dir, l) for l in labels]
        self.files = list()
        self.labels = list()
        self.shuffle = shuffle
        self.sampling_rate = sampling_rate

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
        self.data = None
        self.shapes = self._get_shapes()
        self.load_data()

    def _get_shapes(self):
        shapes = {}
        sample, _ = self._load_sample(0)
        shapes['mfcc'] = list(sample['mfcc'].shape)
        shapes['log_melspectrogram'] = list(sample['log_melspectrogram'].shape)
        return shapes

    def load_data(self):
        start = 0
        length = self._num_examples
        # length = 2000
        if self.shuffle:
            self._shuffle_data()

        mfcc_shape = self.shapes['mfcc']
        mfcc_shape.insert(0, length)
        log_melspectrogram_shape = self.shapes['log_melspectrogram']
        log_melspectrogram_shape.insert(0, length)


        self.data = {
        'mfcc' : np.empty((mfcc_shape), dtype=np.float32),
        'log_melspectrogram' : np.empty((log_melspectrogram_shape), dtype=np.float32),
        }

        y = np.zeros((length), dtype=np.float32)


        for i in range(0, self._num_examples):
            sample, y[i] = self._load_sample(i)
            self.data['mfcc'][i] = sample['mfcc']
            self.data['log_melspectrogram'][i] = sample['log_melspectrogram']

        self.y = y


    @property
    def num_examples(self):
        return self._num_examples

    def _shuffle_data(self):
        np.random.shuffle(self.indices)
        self.files = self.files[self.indices]
        self.labels = self.labels[self.indices]

    def _load_sample(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        raw_audio, _ = librosa.load(file_path, sr=self.sampling_rate)
        sample = preprocess_recording_dict(raw_audio, sr=self.sampling_rate)

        return sample, label
