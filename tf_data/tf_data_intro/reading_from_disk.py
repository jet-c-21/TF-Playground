# import the necessary packages
import os

import numpy as np
import tensorflow as tf
from imutils import paths
from tensorflow.data import AUTOTUNE
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from pyimagesearch.helpers import benchmark


def load_images(imagePath):
    """
    read the image from disk, decode it, resize it,
    and scale the pixels intensities to the range [0, 1]

    :param imagePath:
    :return:
    """
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (96, 96)) / 255.0

    # grab the label and encode it
    label = tf.strings.split(imagePath, os.path.sep)[-2]
    oneHot = label == classNames
    encodedLabel = tf.argmax(oneHot)

    # return the image and the integer encoded label
    return image, encodedLabel


# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
# args = vars(ap.parse_args())
data_dir = 'fruits'

# initialize batch size and number of steps
BS = 64
NUM_STEPS = 1000

# grab the list of images in our dataset directory and grab all unique class names
print("[INFO] loading image paths...")
imagePaths = list(paths.list_images(data_dir))
classNames = np.array(sorted(os.listdir(data_dir)))

# build the dataset and data input pipeline
print("[INFO] creating a tf.data input pipeline..")
dataset = tf.data.Dataset.from_tensor_slices(imagePaths)
dataset = dataset.shuffle(1024).map(load_images, num_parallel_calls=AUTOTUNE).cache().repeat().batch(BS).prefetch(
    AUTOTUNE)

# create a standard image generator object
print("[INFO] creating a ImageDataGenerator object...")
imageGen = ImageDataGenerator(rescale=1.0 / 255)
dataGen = imageGen.flow_from_directory(
    data_dir,
    target_size=(96, 96),
    batch_size=BS,
    class_mode="categorical",
    color_mode="rgb"
)

'''
benchmark the image data generator and display the number of data points generated, 
along with the time taken to perform the operation
'''
totalTime = benchmark(dataGen, NUM_STEPS)
print(f"[INFO] ImageDataGenerator generated {BS * NUM_STEPS} images in {totalTime:.2f} seconds...")

'''
create a dataset iterator, benchmark the tf.data pipeline, 
and display the number of data points generated, along with the time taken
'''
datasetGen = iter(dataset)
totalTime = benchmark(datasetGen, NUM_STEPS)
print(f"[INFO] tf.data generated {BS * NUM_STEPS} images in {totalTime:.2f} seconds...")
