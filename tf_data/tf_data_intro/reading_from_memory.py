# import the necessary packages
from pyimagesearch.helpers import benchmark
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar100
from tensorflow.data import AUTOTUNE
import tensorflow as tf

# initialize the batch size and number of steps
BS = 64
NUM_STEPS = 5000

# load the CIFAR-10 dataset from
print("[INFO] loading the cifar100 dataset...")
(trainX, trainY), (testX, testY) = cifar100.load_data()

# create a standard image generator object
print("[INFO] creating a ImageDataGenerator object...")
imageGen = ImageDataGenerator()
dataGen = imageGen.flow(
    x=trainX, y=trainY,
    batch_size=BS, shuffle=True)

# build a TensorFlow dataset from the training data
dataset = tf.data.Dataset.from_tensor_slices((trainX, trainY))

# build the data input pipeline
print("[INFO] creating a tf.data input pipeline..")
dataset = (dataset
           .shuffle(1024)
           .cache()
           .repeat()
           .batch(BS)
           .prefetch(AUTOTUNE)
           )

# benchmark the image data generator and display the number of data
# points generated, along with the time taken to perform the
# operation
totalTime = benchmark(dataGen, NUM_STEPS)
msg = f"[INFO] ImageDataGenerator generated {BS * NUM_STEPS} images in {totalTime:.2f} seconds..."
print(msg)
pass

# create a dataset iterator, benchmark the tf.data pipeline, and
# display the number of data points generator along with the time taken
datasetGen = iter(dataset)
totalTime = benchmark(datasetGen, NUM_STEPS)
msg = f"[INFO] tf.data generated {BS * NUM_STEPS} images in {totalTime:.2f} seconds..."
print(msg)
