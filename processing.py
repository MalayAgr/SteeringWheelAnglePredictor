import numpy as np
import cv2


vectorized_imread = np.vectorize(cv2.imread,
                                 signature="()->(x,y,z)")
vectorized_imresize = np.vectorize(cv2.resize,
                                   excluded=['dsize', 'interpolation'],
                                   signature='(x,y,z)->(a,b,c)')
vectorized_bgr2yuv = np.vectorize(cv2.cvtColor,
                                  excluded='code',
                                  signature="(x,y,z),()->(a,b,c)")


def preprocess(images, size=(200, 66), epsilon=1e-7):
    images = vectorized_imresize(images,
                                 dsize=size,
                                 interpolation=cv2.INTER_AREA)
    images = vectorized_bgr2yuv(images, cv2.COLOR_BGR2YUV)

    mean = np.mean(images, axis=(1, 2), keepdims=True)
    std = np.std(images, axis=(1, 2), keepdims=True)
    images = (images - mean) / (std + epsilon)

    return images.astype(np.float32)


def augment_images(images, labels):
    to_be_augmented = (np.random.rand(len(images)) < 0.6)

    to_be_flipped = (np.random.rand(len(images)) < 0.5)
    mask = to_be_flipped & to_be_augmented

    images[mask] = np.flipud(images[mask])
    labels[mask] = -labels[mask]
    return images, labels
