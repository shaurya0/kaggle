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
        self.shapes = get_shapes(self.files[0])
        self.load_data()


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

        y = np.empty((length), dtype=np.float32)


        for i in range(0, self._num_examples):
            y[i] = self.labels[idx]
            sample = load_sample(self.files[i])
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

