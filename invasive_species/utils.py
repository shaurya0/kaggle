import os
import shutil
import glob
import numpy as np

def create_labelled_folders(train_folder, labels_csv, label_names):
    train_files = glob.glob(os.path.join(train_folder, '*.jpg'))
    if len(train_files) == 0:
        print("Folders are already labelled")
        return

    labels_df = pandas.read_csv(labels_csv)
    for n in label_names:
        dir_name = os.path.join(train_folder, n)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)


    for x in labels_df.values:
        file_id = x[0]
        label = str(int(x[1]))
        file_name = str(file_id) +'.jpg'
        src = os.path.join(train_folder, file_name)
        dst = os.path.join(train_folder, label, file_name)
        shutil.move(src, dst)


def create_validation_subfolders(train_folder, validation_folder, label_names, split_fraction):
    if not os.path.exists(validation_folder):
        os.mkdir(validation_folder)

    for n in label_names:
        train_sub_folder = os.path.join(train_folder, n)
        if not os.path.exists(train_sub_folder):
            raise FileNotFoundError('training sub folder {} does not exist'.format(train_sub_folder))

        train_files = np.array(glob.glob(os.path.join(train_sub_folder, "*.jpg")))
        num_examples = len(train_files)
        if num_examples == 0:
            raise FileNotFoundError('training sub folder {} does not contain any examples'.format(train_sub_folder))

        validation_sub_folder = os.path.join(validation_folder, n)
        if os.path.exists(validation_sub_folder):
            continue

        os.mkdir(validation_sub_folder)

        randomize = np.arange(num_examples)
        np.random.shuffle(randomize)
        train_files = train_files[randomize]

        validation_size = int(split_fraction*num_examples)
        for fname in train_files[:validation_size]:
            src = fname
            dst = os.path.join(validation_sub_folder, os.path.basename(fname))
            shutil.move(src, dst)
