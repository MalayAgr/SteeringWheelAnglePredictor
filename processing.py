import numpy as np
import cv2


vectorized_imread = np.vectorize(cv2.imread,
                                 signature="()->(x,y,z)")
vectorized_imresize = np.vectorize(cv2.resize,
                                   excluded=['dsize', 'interpolation'],
                                   signature='(x,y,z)->(a,b,c)')
vectorized_cvtColor = np.vectorize(cv2.cvtColor,
                                   excluded='code',
                                   signature="(x,y,z),()->(a,b,c)")


def channelwise_standardization(images, epsilon=1e-7):
    mean = np.mean(images, axis=(1, 2), keepdims=True)
    std = np.std(images, axis=(1, 2), keepdims=True)
    return (images - mean) / (std + epsilon)


def preprocess(
    images, size=(200, 66), epsilon=1e-7, colorspace=cv2.COLOR_BGR2YUV
):
    images = vectorized_imresize(
        images, dsize=size, interpolation=cv2.INTER_AREA
    )
    images = vectorized_cvtColor(images, colorspace)
    images = channelwise_standardization(images, epsilon=epsilon)
    return images.astype(np.float32)


def flip_images(images, labels, mask, threshold=0.5):
    to_be_flipped = (np.random.rand(len(images)) < threshold)
    mask &= to_be_flipped
    images[mask] = np.flipud(images[mask])
    labels[mask] = -labels[mask]
    return images, labels


def augment_images(images, labels, aug_threshold=0.6, flip_threshold=0.5):
    to_be_augmented = (np.random.rand(len(images)) < aug_threshold)
    return flip_images(
        images, labels, to_be_augmented, threshold=flip_threshold
    )
