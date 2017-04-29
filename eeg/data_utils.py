import pickle
from sklearn import preprocessing
import csv
import keras
import numpy as np
import os
import glob

def get_train_files(train_dir_path):
    train_files = glob.glob(os.path.join(train_dir_path, '*_data.csv'))
    train_files.sort()
    return train_files

def get_label_files(train_dir_path):
    label_files = glob.glob(os.path.join(train_dir_path, '*_events.csv'))
    label_files.sort()
    return label_files

def load_data(file_path):
    tmp = np.loadtxt(file_path, delimiter=',', dtype=np.float32, skiprows=1, usecols=tuple(range(1,31)))
    return tmp

def load_labels(file_path):
    tmp = np.loadtxt(file_path, delimiter=',', dtype=np.float32, skiprows=1, usecols=tuple(range(1,6)))
    return tmp

def get_raw_train_data(train_files):
    data_as_list = list()
    num_rows = 0
    for train_file in train_files:
        train_data = load_data(train_file)
        num_rows += train_data.shape[0]
        data_as_list.append(train_data)
    num_cols = data_as_list[0].shape[1]

    data_as_array = np.empty((num_rows, num_cols))

    row_idx = 0
    for train_data in data_as_list:
        end_idx = row_idx + train_data.shape[0]
        data_as_array[row_idx:end_idx,:] = train_data
        row_idx += train_data.shape[0]
    return data_as_array, data_as_list

def pickle_scaler(scaler):
    scaler_file_path = os.path.join('data', 'scaler.pkl')
    with open(scaler_file_path, 'wb') as fid:
        pickle.dump(scaler, fid)

def pickle_obj(obj, file_name):
    file_path = os.path.join('data', file_name)
    with open(file_path, 'wb') as fid:
        pickle.dump(obj, fid)

def unpickle_obj(file_name):
    file_path = os.path.join('data', file_name)
    if not os.path.isfile(file_path):
        raise FileNotFoundError('{} not found'.format(file_path))

    with open(file_path, 'rb') as fid:
        return pickle.load(fid)
