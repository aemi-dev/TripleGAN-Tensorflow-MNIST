'''
MNIST.PY
'''

import numpy as np
from collections import defaultdict
from keras.datasets import mnist

class_num = 10
image_size = 28
img_channels = 1

'''
Shuffle Datas and prepare for a given number of labels
'''
def prepare_data(n):
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data, test_data = color_preprocessing(train_data, test_data) # pre-processing

    criteria = n//10
    input_dict, labelled_x, labelled_y, unlabelled_x, unlabelled_y = defaultdict(int), list(), list(), list(), list()

    # make pairs with labels and datas
    for image, label in zip(train_data,train_labels) :
        if input_dict[int(label)] != criteria :
            input_dict[int(label)] += 1
            labelled_x.append(image)
            labelled_y.append(label)

        unlabelled_x.append(image)
        unlabelled_y.append(label)


    labelled_x = np.asarray(labelled_x)
    labelled_y = np.asarray(labelled_y)
    unlabelled_x = np.asarray(unlabelled_x)
    unlabelled_y = np.asarray(unlabelled_y)

    print("labelled data:", np.shape(labelled_x), np.shape(labelled_y))
    print("unlabelled data :", np.shape(unlabelled_x), np.shape(unlabelled_y))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(labelled_x))
    labelled_x = labelled_x[indices]
    labelled_y = labelled_y[indices]

    indices = np.random.permutation(len(unlabelled_x))
    unlabelled_x = unlabelled_x[indices]
    unlabelled_y = unlabelled_y[indices]

    print("======Prepare Finished======")


    labelled_y_vec = np.zeros((len(labelled_y), 10), dtype=np.float)
    for i, label in enumerate(labelled_y) :
        labelled_y_vec[i, labelled_y[i]] = 1.0

    unlabelled_y_vec = np.zeros((len(unlabelled_y), 10), dtype=np.float)
    for i, label in enumerate(unlabelled_y) :
        unlabelled_y_vec[i, unlabelled_y[i]] = 1.0

    test_labels_vec = np.zeros((len(test_labels), 10), dtype=np.float)
    for i, label in enumerate(test_labels) :
        test_labels_vec[i, test_labels[i]] = 1.0


    return labelled_x, labelled_y_vec, unlabelled_x, unlabelled_y_vec, test_data, test_labels_vec


'''
Normalization
'''
def color_preprocessing(x_train, x_test):
    x_train = x_train/127.5 - 1
    x_test = x_test/127.5 - 1
    return x_train, x_test