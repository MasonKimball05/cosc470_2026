# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 11:45:47 2021

@author: karto
"""

from util import evaluateConfidence, evaluateAccuracy
import numpy as np
import network2
import mnist_loader
import matplotlib.pyplot as plt

# uncomment these lines if you are using a mac
import theano
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

def predictDigitNeuralNetwork(img):
    a = net.feedforward(img)
    return np.argmax(a)

def convertAccuracies(accuracies,datalength):
    return [epochacc/datalength for epochacc in accuracies]

mini_batch_size = 10

# solution 1 to the "overfitting problem" - use validation_data to "see" when accuracy has saturated on the validation_data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, test_cost, test_accuracy = net.SGD(training_data, 50, 10, 0.5, evaluation_data=validation_data,monitor_training_accuracy=True,monitor_evaluation_accuracy=True,test_data=test_data,early_stopping_n=10)
fig, ax = plt.subplots()
mega_fig, mega_ax = plt.subplots()
xvals = range(len(evaluation_accuracy))
yvals_trainingdata = convertAccuracies(training_accuracy, len(training_data))
yvals_validationdata = convertAccuracies(evaluation_accuracy, len(validation_data))
yvals_testdata = convertAccuracies(test_accuracy, len(test_data))
ax.plot(xvals,yvals_trainingdata)
ax.plot(xvals,yvals_validationdata)
ax.plot(xvals,yvals_testdata)
mega_ax.plot(xvals,yvals_testdata)
ax.legend(["training_data accuracy", "validation_data accuracy", "test_data accuracy"])
ax.set_title("SOL1: EARLY STOPPING")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")

# solution 2 to the "overfitting problem" - expand the training dataset
expanded_training_data, validation_data, test_data = mnist_loader.load_data_wrapper("mnist_expanded.pkl.gz")
expanded_training_data = list(expanded_training_data)
validation_data = list(validation_data)
test_data = list(test_data)
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, test_cost, test_accuracy = net.SGD(expanded_training_data, 50, 10, 0.5, evaluation_data=validation_data,monitor_training_accuracy=True,monitor_evaluation_accuracy=True,test_data=test_data)
xvals = range(len(evaluation_accuracy))
yvals_trainingdata = convertAccuracies(training_accuracy, len(expanded_training_data))
yvals_validationdata = convertAccuracies(evaluation_accuracy, len(validation_data))
yvals_testdata = convertAccuracies(test_accuracy, len(test_data))
fig, ax = plt.subplots()
ax.plot(xvals,yvals_trainingdata)
ax.plot(xvals,yvals_validationdata)
ax.plot(xvals,yvals_testdata)
mega_ax.plot(xvals,yvals_testdata)
ax.legend(["training_data accuracy", "validation_data accuracy", "test_data accuracy"])
ax.set_title("SOL2: EXPANDED TRAINING DATA")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")

# solution 3 - regularization
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy, test_cost, test_accuracy = net.SGD(training_data, 50, mini_batch_size, 0.5, evaluation_data=validation_data,monitor_training_accuracy=True, monitor_evaluation_accuracy=True, test_data=test_data, lmbda=0.1)
xvals = range(len(evaluation_accuracy))
yvals_trainingdata = convertAccuracies(training_accuracy, len(training_data))
yvals_validationdata = convertAccuracies(evaluation_accuracy, len(validation_data))
yvals_testdata = convertAccuracies(test_accuracy, len(test_data))
fig, ax = plt.subplots()
ax.plot(xvals,yvals_trainingdata)
ax.plot(xvals,yvals_validationdata)
ax.plot(xvals,yvals_testdata)
mega_ax.plot(xvals,yvals_testdata)
ax.legend(["training_data accuracy", "validation_data accuracy", "test_data accuracy"])
ax.set_title("SOL3: REGULARIZATION lmbda 0.1")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")

# all three solutions applied together
# do one more fig, ax plot and then one more mega_ax.plot



mega_ax.legend(["sol1", "sol2", "sol3", "sol4"])
mega_ax.set_title("OVERALL COMPARISON OF TEST_DATA ACCURACY")
mega_ax.set_xlabel("epoch")
mega_ax.set_ylabel("accuracy")
