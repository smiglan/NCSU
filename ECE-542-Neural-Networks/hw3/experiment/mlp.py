#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pdb
sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = '../../data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784,20, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)

def main():

    train_data, valid_data, test_data = load_data()


    model = network2.Network([784,80,10])
    # train the network using SGD
    e_cost, e_accuracy, t_cost, t_accuracy = model.SGD(
        training_data=train_data,
        epochs=100,
        mini_batch_size=16,
        eta=6e-4,
        lmbda = 0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
    num_correct = 0

    
    plt.plot(list(range(0,len(t_accuracy))),[x/len(train_data[0]) for x in t_accuracy], label='Training')
    plt.plot(list(range(0,len(e_accuracy))),[x/len(valid_data[0]) for x in e_accuracy], label='Validation')
    legend = plt.legend(loc='upper left', shadow=True)
    plt.title("Learning curve for Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy')
    plt.clf()

    plt.plot(list(range(0,len(t_cost))),t_cost,label='Training')
    plt.plot(list(range(0,len(e_cost))),e_cost,label='Validation')
    legend = plt.legend(loc='upper left', shadow=True)
    plt.title(" Learning curve for Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss')
    plt.clf()

    for i in range(len(test_data[0])):
        prediction = np.argmax(model.feedforward(test_data[0][i]))
        if prediction == test_data[1][i]:
            num_correct += 1
    test_accuracy = num_correct / len(test_data[0])
    print("Test accuracy: " + str(test_accuracy))

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
