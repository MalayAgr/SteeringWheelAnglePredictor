# Building a Self-Driving Car Using Deep Learning and Python

[Subham Dasgupta](https://github.com/Subhamdg) and I set out to build a model which is capable of predicting the steering angle for a car given an image of the road as input, as part of our summer internship project at [MyWBUT](https://www.mywbut.com/).

This document highlights the steps we took to achieve that goal.

## Table of Contents

<!-- MarkdownTOC -->

- General Approach
    - The Code
    - The Model
    - The Data
- The "API"
    - Loading Data \(Module: data\)
        - data.flatten_csv\(path, data_dir, column_names, header=None, usecols=&#91;0, 1, 2, 3&#93;, shift=0.2\)
            - Example
    - Preprocessing Data \(Module: processing\)
        - processing.vectorized_imread\(image_paths\)
        - processing.vectorized_imresize\(images, dsize, interpolation=cv2.INTER_LINEAR\)
        - processing.vectorized_cvtColor\(images, code\)
        - processing.channelwise_standardization\(images, epsilon=1e-7\)
        - processing.preprocess\(images, size=(200, 66\), epsilon=1e-7, colorspace=cv2.COLOR_BGR2YUV)
    - Augmenting Data \(Module: processing\)
        - processing.augment_images\(images, labels, aug_threshold=0.6, flip_threshold=0.5\)
        - processing.flip_images\(images, labels, mask=None, threshold=0.5\)
    - Building the Model \(Module: model\)
        - model.activation_layer\(ip, activation\)
        - model.conv2D\(ip, filters, kernel_size, strides, layer_num, activation, kernel_initializer='he_uniform', bias_val=0.01\)
- References

<!-- /MarkdownTOC -->

## General Approach

### The Code

Our approach was to make the code as modular as possible, thereby making customization to the model as easy as possible. With that in mind, we attempted to create an API-like code structure where different modules can call functions of other modules without exactly being aware of how those functions are implemented. All in all, our project is divided into the following four files:

1. [data.py](data.py) - Responsible for all data-related tasks such as loading, cleaning, etc.
2. [processing.py](processing.py) - Responsible for all preprocessing related tasks such as resizing, cropping, color conversions, augmentation, etc
3. [model.py](model.py) - Responsible for building and training the model
4. [main.py](main.py) - Main driver program which uses the above modules to obtain a trained model

We used [TensorFlow's Keras API](https://keras.io/) to build the model, strictly adhering to the functional API for greater flexibility. [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [OpenCV](https://github.com/skvark/opencv-python) and [scikit-learn](https://scikit-learn.org/) were used for data processing.

The code will be covered in greater details in the following sections.

### The Model

Our approach was based on a seminal paper published by NVIDIA in 2016 [[1]](#ref-1). The paper suggests using a Convolutional Neural Network (CNN) model trained on images of roads with the steering angle as a training signal, learning to predict the angle when given new images of roads.

The architecture of the CNN model is as follows:

* A `3 X 66 X 200` input layer, normalized or standardized
* A convolutional layer with `24` kernels, `5 X 5` in size and stride of `2`
* A convolutional layer with `36` kernels, `5 X 5` in size and stride of `2`
* A convolutional layers with `48` kernels, `5 X 5` in size and stride of `2`
* Two convolutional layers with `64` kernels each, `3 X 3` in size and stride of `1`
* Three fully-connected layers with `100`, `50` and `10` units respectively
* A final fully-connected layer with `1` unit for the output

<div align="center" style="padding: 10px;">
    <img src="img/model_diagram.png" width="500" height="500" alt="Model">
    <div>
        <em>Source: <a href="#ref-1">[1]</a></em>
    </div>
</div>

This leads to ~2,50,000 trainable parameters in the model.

While we have followed this architecture closely, we have experimented with various activation functions and have also designed the code such that it is extremely convenient to switch the activation function for the entire model. Specifically, in addition to using the standard **ReLU** activation with He Normal initialization and a small constant bias, we have experimented with **ELU** [[2]](#ref-2) and **LeakyReLU** [[3]](#ref-3) as our activation functions.

ELU helps with dealing with dying neurons and exploding gradients better than ReLU and in the end, became the activation function of choice for our model. LeakyReLU was found to perform worse than both ELU and ReLU, and now exists only as an API feature.

In addition to this, we have also added some modern-day conventional practices to the model. Specifically, the following two practices have been adopted:

* **Bath Normalization** - Batch Normalization is added after each convolutional layer to improve convergence speed and to combat overfitting
* **Dropout** [[4]](#ref-4) - Dropout is added between the convolutional block and the fully-connected block to combat overfitting

### The Data

[[1]](#ref-1) used a physical vehicle mounted with cameras to take images of roads and to record steering angles as the vehicle is driven. They are NVIDIA and we are college students. Clearly, this sort of luxury wasn't available to us. Instead, we used Udacity's open-source [self-driving car simulator](https://github.com/udacity/self-driving-car-sim) to generate data to train the model.

The simulator comes with two modes of operation, training and autonomous:

* **Training Mode** - In this mode, a human is supposed to drive a car on a track. In the background, the simulator will automatically take images of the road and record the corresponding steering angle, among other things. Rather than taking one picture for each angle, the simulator takes 3 images from three different angles (center, left, right) for each angle. This gives your model the ability to learn the angle for different views of the same road and thus, leads to more accuracy. Though three images are taken, only one steering angle is recorded, from the perspective of the center angle.
    
    The final output is a CSV file with `m` rows (number of steering angles) and  7 columns. The first three columns correspond to the file paths of the center, left and right images for the angle. The fourth column corresponds to the steering angle. The last three columns correspond to throttle, reverse and speed (a model can be built which predicts these values as well) respectively.

* **Autonomous Mode** - In this mode, the simulator uses the trained model saved as an `.h5` file to obtain predicted steering angles and drive the car.

It is also equipped with two tracks and more tracks can be added with a bit of hacking (refer to link above).

Our model was trained to be able to drive on the two default tracks available in the simulator.

## The "API"

This section documents the complete API available to the user for processing data, building the model and training it.

> **NOTE:** Having this API is not enough to actually enable the simulator to drive the car using the model. There needs to be a script in between which relays data to and fro between the model and the simulator.
> 
### [Loading Data (Module: data)](data.py)

#### [data.flatten_csv(<em>path, data_dir, column_names, header=None, usecols=&#91;0, 1, 2, 3&#93;, shift=0.2</em>)](/blob/34977001664d516b1e2ae007ddc3c0bebf2da39a/data.py#L6)

"Flattens" the CSV files so that the the three columns containing the three angles become rows in themselves and the steering angle gets repeated for them. It also shifts the steering angle for right (subtracting `shift`) and left images (adding `shift`) as all angles are with respect to center image.

As stated above, the simulator outputs a CSV file where each row has paths to images from three different angles and the steering angle. A standard CNN can take as input only one image at a time and therefore, for each row, we are forced to select one of the three angles. While a good model can be obtained by randomly selecting any one of the angles, a better model can be obtained by feeding it all the angles.

Effectively, if your CSV file has `m` rows, the function will give you 1D arrays of size `3m`.

| **Arguments**         |                                                                                                                                                                                           |
|----------------   |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   |
| `path`            |                                                                                `str`: Path to CSV file.                                                                                   |
| `data_dir`        |                     `str`: Directory where the images are stored. This is used to make the paths in the CSV file absolute so that there are no data opening errors.                       |
| `column_names`    |                           `list`: The names that should be used for the columns in CSV file. Last item should always be for the column containing the labels.                             |
| `header`          |                                                  `bool`: Indicates whether the CSV file contains a header or not. Defaults to `None`.                                                     |
| `usecols`         | `list`: The index of columns to be used in the CSV. Last item should always be for the column containing the labels. Defaults to `[0, 1, 2, 3]`, using first four columns of the file.    |
| `shift`           | `float`: The amount by which the right and left images will be shifted. Defaults to `0.2`.                                                                                                |
| **Returns**   |                                                                           |
| `images`      | `numpy.array`: A 1D numpy array with the flattened image paths            |
| `labels`      | `numpy.array`: A 1D numpy array with the flattened and shifted labels     |

##### Example

Say this is our CSV file:

| img_center.png    | img_left.png  | img_right.png     | 0.5   |
|----------------   |-------------- |---------------    |-----  |

Calling the function on this file will return:

```python
 images = ['img_center.png', 'img_left.png', 'img_right.png']
 labels = [0.5, 0.7, 0.3]
```


----

### [Preprocessing Data (Module: processing)](processing.py)

#### [processing.vectorized_imread(image_paths)](https://github.com/MalayAgarwal-Lee/steering_wheel_angle/blob/eaf8f4b97bc57dddf6ba77ef0a9a919a7b6e34c0/processing.py#L5)

A vectorized implementation of OpenCV's `imread()` function developed using `numpy.vectorize()`, used to obtain a 4D array of images from a list of image paths in a single call.

Arguments are same as the normal `imread()` function, except that this function takes multiple image paths instead of a single image path.

It returns the opened images as a 4D array.


#### [processing.vectorized_imresize(images, dsize, interpolation=cv2.INTER_LINEAR)](https://github.com/MalayAgarwal-Lee/steering_wheel_angle/blob/c30377f8a199db07d56d880c18c7acfb7524675e/processing.py#L7)

A vectorized implementation of OpenCV's `resize()` function developed using `numpy.vectorize()`, used to resize a bunch of images in a single call.

Arguments are same as the normal `resize()` function, except that this function takes multiple images instead of a single image.

It returns the resized images as a 4D array.

> **NOTE:** This doesn't have, most of the times, any performance gains. The internal implementation is essentially a loop. It exists purely for conciseness.


#### [processing.vectorized_cvtColor(images, code)](https://github.com/MalayAgarwal-Lee/steering_wheel_angle/blob/c30377f8a199db07d56d880c18c7acfb7524675e/processing.py#L10)

A vectorized implementation of OpenCV's `cvtColor()` function developed using `numpy.vectorize()`, used to change the colorspace of a bunch of images in a single call.

Arguments are same as the normal `cvtColor()` function, except that this function takes multiple images instead of a single image.

It returns the converted images as a 4D array.

> **NOTE:** This doesn't have, most of the times, any performance gains. The internal implementation is essentially a loop. It exists purely for conciseness.

#### [processing.channelwise_standardization(<em>images, epsilon=1e-7</em>)](https://github.com/MalayAgarwal-Lee/steering_wheel_angle/blob/7e6fd9817bb41c04198a70818fd97761d532ded6/processing.py#L25)

Standardizes the images so that they have 0 mean and unit standard deviation per channel (RGB, YUV, etc.)

| **Arguments**     |                                                                                                                           |
|---------------    |------------------------------------------------------------------------------------------------------------------------   |
| `images`          | `numpy.array`: A 4D array of images as (N X height x width X channels)                                                    |
| `epsilon`         | `float`: A small value that will be added to the standard deviation to prevent `ZeroDivisionError`. Defaults to `1e-7`    |
| **Returns**   |                                                   |
| `images`      | `numpy.array`: The standardized images as a 4D array   |

#### [processing.preprocess(<em>images, size=(200, 66), epsilon=1e-7, colorspace=cv2.COLOR_BGR2YUV</em>)](https://github.com/MalayAgarwal-Lee/steering_wheel_angle/blob/0a39d7c5d90c23d28e69c0c3eba73bf9a6b59f15/processing.py#L21)

Combines resizing, colorspace conversion and standardization into one single function, becoming the preprocessor in the pipeline. It uses `processing.vectorized_imresize()`, `processing.vectorized_cvtColor()` and `processing.channelwise_standardization()` to perform these tasks. Additionally, the function changes the data type used for the images from `float64` (NumPy's default) to `float32` to reduce amount of memory occupied by the images.

| **Arguments**     |                                                                                       |
|---------------    |-------------------------------------------------------------------------------------  |
| `images`          | `numpy.array`: A 4D array of images as (N X height x width X channels)                |
| `size`            | `tuple`: The target size of the images as (width X height). Defaults to `(200, 66)`   |
| `epsilon`         | `float`: A small value that will be added to the standard deviation to prevent `ZeroDivisionError`. Defaults to `1e-7`    |
| `colorspace`      | `cv2.ColorConversionCode`: A code representing the target colorspace. Defaults to `cv2.COLOR_BGR2YUV`     |
| **Returns**   |                                                   |
| `images`      | `numpy.array`: The preprocessed images as a 4D array   |


----

### [Augmenting Data (Module: processing)](processing.py)

#### [processing.augment_images(<em>images, labels, aug_threshold=0.6, flip_threshold=0.5</em>)](https://github.com/MalayAgarwal-Lee/steering_wheel_angle/blob/0a39d7c5d90c23d28e69c0c3eba73bf9a6b59f15/processing.py#L40)

Augments images according to the given threshold. It currently supports only random left/right flipping according to the given flip threshold.

| **Arguments**     |                                                                                           |
|------------------ |----------------------------------------------------------------------------------------   |
| `images`          | `numpy.array`: A 4D array of images as (N X height x width X channels).                   |
| `labels`          | `numpy.array`: A 1D array containing all the labels for the images.                       |
| `aug_threshold`   | `float`: The minimum probability for an image to not be augmented. Defaults to `0.6`.     |
| `flip_threshold`  | `float`: The minimum probability for an image to not be flipped. Defaults to `0.5`.       |
| **Returns**   |                                                                   |
| `images`      | `numpy.array`: Augmented and unaugmented images as a 4D array.    |
| `labels`      | `numpy.array`: Augmented and unaugmented labels as a 1D array.    |

#### [processing.flip_images(<em>images, labels, mask=None, threshold=0.5</em>)](https://github.com/MalayAgarwal-Lee/steering_wheel_angle/blob/0a39d7c5d90c23d28e69c0c3eba73bf9a6b59f15/processing.py#L32)

A helper function which flips the images left/right that are outside the threshold *and* `True` in `mask` if it is specified, flipping the corresponding labels as well (simple negation).

`processing.augment_images()` calls this function and passes `mask` based on `aug_threshold` to ensure that flipped images are among the images that are to be augmented.

| **Arguments**     |                                                                                                                                                       |
|---------------    |---------------------------------------------------------------------------------------------------------------------------------------------------    |
| `images`          | `numpy.array`: A 4D array of images as (N X height x width X channels).                                                                               |
| `labels`          | `numpy.array`: A 1D array containing all the labels for the images.                                                                                   |
| `mask`            | `numpy.array`: A Boolean mask which can be used to apply another condition to determine whether flipping will be done or not. Defaults to `None`.     |
| `threshold`       | `float`: The minimum probability for an image to not be flipped. Defaults to `0.5`.                                                                   |
| **Returns**   |                                                                   |
| `images`      | `numpy.array`: Flipped and normal images as a 4D array.    |
| `labels`      | `numpy.array`: Flipped and normal labels as a 1D array.    |


----

### [Building the Model (Module: model)](model.py)

#### [model.activation_layer(<em>ip, activation</em>)](https://github.com/MalayAgarwal-Lee/steering_wheel_angle/blob/eaf8f4b97bc57dddf6ba77ef0a9a919a7b6e34c0/model.py#L8)

Initializes a `ReLU`, `ELU` or `LeakyReLU` activation layer with the given input layer based on `activation`. This function can be used in place of the `activation` keyword argument in all Keras layers to mix-match activations for different layers and easily use `ELU`, `LeakyReLU`, which otherwise need to be imported separately.

| **Arguments**     |                                                                                                                                       |
|---------------    |-------------------------------------------------------------------------------------------------------------------------------------  |
| `ip`              | `keras.layers.Layer`: Any Keras layer such as `Input`, `Conv2D`, `Dense`, etc., which will be used as the input for the activation.   |
| `activation`      | `str`: The activation layer to be used. Can be `relu`, `elu` or `lrelu`.                                                              |
| **Returns**       |                                                                                                                                       |
| `activation`      | `keras.layers.Layer`: Either of `ReLU`, `ELU` or `LeakyReLU` activation layers initialized with the given input.                                  |
| **Raises**        |                                                                                                                                       |
| `KeyError`        | When `activation` is not one of the specified values.                                                                                 |

#### [model.conv2D(<em>ip, filters, kernel_size, strides, layer_num, activation, kernel_initializer='he_uniform', bias_val=0.01</em>)](https://github.com/MalayAgarwal-Lee/steering_wheel_angle/blob/eaf8f4b97bc57dddf6ba77ef0a9a919a7b6e34c0/model.py#L14)

Initializes a "convolutional block" which has a Conv2D layer initialized with the given input, a BatchNormalization layer and an activation layer determined by `activation`.


| **Arguments**         |                                                                                                                                                                                       |
|---------------------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  |
| `ip`                  | `keras.layers.Layer`: Any Keras layer such as `Input`, `Conv2D`, `Dense`, etc., which will be used as the input for the `Conv2D` layer.                                               |
| `filters`             | `int`: The number of filters to be used in the `Conv2D` layer.                                                                                                                        |
| `kernel_size`         | `tuple`: The size of each filter in the `Conv2D` layer.                                                                                                                               |
| `strides`             | `strides`: The size of the stride for each filter in the `Conv2D` layer.                                                                                                              |
| `layer_num`           | `int`: An index value for the layer which will be used in the naming of the layer.                                                                                                    |
| `activation`          | `str`: The activation layer to be used. Can be `relu`, `elu` or `lrelu`.                                                                                                              |
| `kernel_initializer`  | `str`: The weight initializer for each filter. Defaults to `he_uniform`.                                                                                                              |
| `bias_val`            | `float`: The initial bias value to be used for each filter. Defaults to `0.01`.                                                                                                       |
| **Returns**           |                                                                                                                                                                                       |
| `conv2dlayer`         | `keras.layers.Layer`: A Keras layer composed of a `Conv2D` layer, a `BatchNormalization` layer and an activation layer, where the Conv2D layer is initialized with the given input.   |


## References

* <a  id="ref-1">[1]</a> Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, and Karol Zieba. (2016). [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316).
* <a  id="ref-2">[2]</a> Djork-Arné Clevert, Thomas Unterthiner, & Sepp Hochreiter. (2015). [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289).
* <a  id="ref-3">[3]</a> Andrew L. Maas. (2013). [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf).
* <a  id="ref-4">[4]</a> Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, & Ruslan Salakhutdinov (2014). [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/v15/srivastava14a.html). Journal of Machine Learning Research, 15(56), 1929-1958.






