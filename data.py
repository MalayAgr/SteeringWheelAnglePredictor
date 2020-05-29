import pandas as pd
import numpy as np
import os


def flatten_csv(path,
                data_dir,
                column_names,
                header=None,
                usecols=[0, 1, 2, 3]):

    df = pd.read_csv(path,
                     header=header,
                     names=column_names,
                     usecols=usecols)

    images = df[column_names[:3]].values
    labels = df[column_names[-1]].values

    shape = (images.shape[0] * images.shape[1],)
    images = images.reshape(shape)

    labels = np.repeat(labels, repeats=3)

    df = pd.DataFrame(data=images, columns=['image'])

    df['steer'] = labels

    df.loc[df['image'].str.contains('right'), 'steer'] = df['steer'] - 0.2
    df.loc[df['image'].str.contains('left'), 'steer'] = df['steer'] + 0.2

    df['image'] = os.path.abspath(data_dir) + df['image']

    return df['image'].values, df['steer'].values
