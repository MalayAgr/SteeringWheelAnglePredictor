import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split


def flatten_csv(
    path, data_dir, column_names, header=None, usecols=[0, 1, 2, 3],
    shift=0.2
):
    df = pd.read_csv(
        path, header=header, names=column_names, usecols=usecols
    )

    images = df[column_names[:3]].values
    labels = df[column_names[-1]].values

    shape = (images.shape[0] * images.shape[1],)
    images = images.reshape(shape)

    labels = np.repeat(labels, repeats=3)

    df = pd.DataFrame(data=images, columns=['image'])

    df['steer'] = labels

    df.loc[df['image'].str.contains('right'), 'steer'] = df['steer'] - shift
    df.loc[df['image'].str.contains('left'), 'steer'] = df['steer'] + shift

    df['image'] = os.path.abspath(data_dir) + df['image']

    return df['image'].values, df['steer'].values


def load_and_split_data(
    path, data_dir, column_names, test_size=.15, val_size=.15
):
    images, labels = flatten_csv(
        path=path, data_dir=data_dir, column_names=column_names
    )

    im_train, im_test, labels_train, labels_test = train_test_split(
        images, labels, test_size=test_size, shuffle=True
    )
    im_train, im_val, labels_train, labels_val = train_test_split(
        im_train, labels_train, test_size=val_size, shuffle=True
    )
    return im_train, im_val, im_train, labels_train, labels_val, labels_test
