"""
Created by: Topiltzin Hernández Mares
Date: 19/08/2021
Model source: https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/
"""

from typing import List
from os import getcwd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LinearRegressionGD(object):
    def __init__(self, learning_rate: float = 0.05, iterations: int = 1000) -> None:
        """
        :param learning_rate:
            1. If we choose the value to be very large, GD can overshoot the minimum.
            If may fail to coverage or even diverge.
            2. If we choose the value to be very small, GD will take small steps to
            reach local minima and will take a longer time to reach minima.
        :param iterations: Number of passes over the training set.
        """

        self._learning_rate = learning_rate
        self._iterations = iterations
        self._initial_cost: List[float] = []
        self._cost = self._initial_cost
        self._weights: np.ndarray = np.ndarray((0, 0))

    def train(self, samples: np.ndarray, values: np.ndarray) -> None:
        """Train the model with data

        :param samples: Training samples
        :param values: Training values
        :return: None
        """

        # init the values
        self._cost = self._initial_cost
        self._weights = np.zeros((samples.shape[1], 1))
        # get the dimensions of the samples
        m = samples.shape[0]

        # fit the weights of the hypothesis and calculate cost
        for _ in range(self._iterations):
            # calculate the hypothesis function and get predicted values
            predicted = np.dot(samples, self._weights)
            residuals = predicted - values
            # could calculate the partial derivatives iterative,
            # but with linear algebra we do it in "one step"
            gradient_vector = np.dot(samples.T, residuals)
            # recalculate hypothesis weights
            self._weights -= (self._learning_rate / m) * gradient_vector
            # calculate new cost
            cost = np.sum((residuals ** 2)) / (2 * m)
            self._cost.append(cost)

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """
        Predicts values with a trained model
        :param samples: The samples to predict
        :return: The predicted values
        """
        return np.dot(samples, self._weights)


if __name__ == "__main__":
    # init the plot figures
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # get data for the model
    data = pd.read_csv(f"{getcwd()}/insurance.csv")
    filtered_data = data[~data["smoker"].str.contains("no")]

    # split train and test data
    data_half_length = len(filtered_data.index) // 2

    train_data = filtered_data[:data_half_length]
    test_data = filtered_data[data_half_length:]

    # get train x and y values
    bmi_train = train_data[["bmi"]].values * 1e-2
    charges_train = train_data[["charges"]].values * 1e-5

    # get the test x values
    bmi_test = test_data[["bmi"]].values * 1e-2
    charges_test = test_data[["charges"]].values * 1e-5

    # train the model
    model = LinearRegressionGD()
    model.train(bmi_train, charges_train)

    # test the model
    predicted_values = model.predict(bmi_test)

    ax1.scatter(bmi_train, charges_train, s=10, c="b", label="Transaction date train")
    ax1.scatter(bmi_test, charges_test, s=10, c="g", label="Transaction date test")
    ax1.plot(bmi_test, predicted_values, c="r", label="predicted", linewidth=3)
    plt.show()
