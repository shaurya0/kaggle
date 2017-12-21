import librosa

from os.path import join as path_join
from os import mkdir
from os.path import isdir
import glob

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
