""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


def display_digits():
    digits = load_digits()
    print(digits.DESCR)
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i+1)
        subplot.matshow(numpy.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()


def train_model():
    data = load_digits()
    num_trials = 500
    train_percentages = list(range(5, 95, 5))
    test_accuracies = numpy.zeros(len(train_percentages))
    index = 0

    for i in train_percentages:
        accuracy_total = 0
        for trial in range(num_trials):
            X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                                data.target,
                                                                train_size=i)
            model = LogisticRegression(C=4**-3)
            model.fit(X_train, y_train)

            accuracy_total = accuracy_total + (model.score(X_test, y_test))
            # print(model.score(X_test, y_test))

        test_accuracies[index] = accuracy_total / num_trials
        index += 1
    print(test_accuracies)
    print(train_percentages)

    # train models with training percentages between 5 and 90 (see
    # train_percentages) and evaluate the resultant accuracy for each.
    # You should repeat each training percentage num_trials times to smooth out
    # variability.
    # For consistency with the previous example use
    # model = LogisticRegression(C=10**-10) for your learner

    fig = plt.figure()
    plt.plot(train_percentages, test_accuracies)
    plt.xlabel('Percentage of Data Used for Training')
    plt.ylabel('Accuracy on Test Set')
    plt.show()


if __name__ == "__main__":
    # display_digits()
    train_model()
