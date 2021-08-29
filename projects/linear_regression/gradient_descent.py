"""
Created by: Topiltzin Hernández Mares
Date: 19/08/2021
Model source: https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/
"""

from typing import List

import numpy as np


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

    @staticmethod
    def get_mean_squared_error(predicted_values: np.ndarray, real_values: np.ndarray) -> float:
        error = 0.0

        if len(predicted_values) != len(real_values):
            raise Exception("Predicted values and real values are not of the same size")

        for predicted, real in np.c_[predicted_values, real_values]:
            error += (real - predicted) ** 2

        return error / len(predicted_values)

    @staticmethod
    def get_r2(predicted_values: np.ndarray, real_values: np.ndarray) -> float:
        rss = 0.0
        for predicted, real in np.c_[predicted_values, real_values]:
            rss += (real - predicted) ** 2

        real_values_mean = np.mean(real_values)
        tss = 0.0
        for value in real_values:
            tss += (float(value) - real_values_mean) ** 2

        return 1 - (rss / tss)
