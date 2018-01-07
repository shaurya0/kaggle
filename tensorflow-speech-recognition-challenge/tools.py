import librosa

from os.path import join as path_join
from os import mkdir
from os.path import isdir
import glob
import numpy as np

from random import shuffle
from shutil import move as file_move

SAMPLING_RATE = 16000

def create_silence_recordings(input_files, output_folder, sr=SAMPLING_RATE, stride=SAMPLING_RATE, duration=SAMPLING_RATE):
    if not isdir(output_folder):
        mkdir(output_folder)

    for (i,f) in enumerate(input_files):
        y,sr = librosa.load(f, sr=sr)
        for j in range(0, len(y), stride):
            output_path = path_join(output_folder, '{}_{}.wav'.format(i,j))
            librosa.output.write_wav(output_path, y[j:j+duration], sr=SAMPLING_RATE)


def split_train_validation(labels, split_fraction, input_dir, output_dir):
    if not isdir(output_dir):
        mkdir(output_dir)

    for label in labels:
        subfolder = path_join(input_dir, label)
        wav_files = glob.glob(path_join(subfolder, '*.wav'))
        num_wav_files = len(wav_files)
        num_files_to_move = int(split_fraction * num_wav_files)
        print('moving {} files with label {} to validation folder'.format(num_files_to_move, label))
        shuffle(wav_files)

        wav_files_to_move = wav_files[:num_files_to_move]
        output_subfolder = path_join(output_dir, label)
        if not isdir(output_subfolder):
            mkdir(output_subfolder)
        for wav_file in wav_files_to_move:
            output_file = path_join(output_subfolder, basename(wav_file))
            file_move(wav_file, output_file)

def preprocess_recording(x, sr):
    length = len(x)
    if length < sr:
        zero_pad_length = sr - length
        tmp = np.zeros((zero_pad_length, ), dtype=np.float32)
        x = np.concatenate((tmp, x), axis=0)

    melspectrogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=128, fmax=8000, hop_length=512)
    log_melspectrogram = librosa.core.logamplitude(melspectrogram)
    log_melspectrogram -= np.mean(log_melspectrogram, axis=0)
    log_melspectrogram /= (np.std(log_melspectrogram) + 1e-6)

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melspectrogram), n_mfcc=40)
    mfcc -= np.mean(mfcc, axis=0)
    mfcc /= (np.std(mfcc) + 1e-6)

    return {"log_melspectrogram" : log_melspectrogram[:, :, np.newaxis], "mfcc" : mfcc[:, :, np.newaxis]}


def load_sample(file_path):
    raw_audio, _ = librosa.load(file_path, sr=self.sampling_rate)
    sample = preprocess_recording(raw_audio, sr=self.sampling_rate)

    return sample

def get_shapes(example_filepath):
    shapes = {}
    sample, _ = load_sample(example_filepath)
    shapes['mfcc'] = list(sample['mfcc'].shape)
    shapes['log_melspectrogram'] = list(sample['log_melspectrogram'].shape)
    return shapes
