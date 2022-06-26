# coding: utf-8
"""
Author: Jet C.
GitHub: https://github.com/jet-c-21
Create Date: 2022/6/20
"""
import time


def benchmark(datasetGen, numSteps):
    # start our timer
    start = time.time()
    # loop over the provided number of steps
    for i in range(0, numSteps):
        # get the next batch of data (we don't do anything with the
        # data since we are just benchmarking)
        images, labels = next(datasetGen)

    # stop the timer
    end = time.time()

    # return the difference between end and start times
    return end - start


