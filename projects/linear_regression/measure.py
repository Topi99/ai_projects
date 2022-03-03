import time
from os import getcwd
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projects.linear_regression.gradient_descent import LinearRegressionGD


DataType = Dict[str, np.ndarray]
StatsType = Dict[str, float]


def get_data_old() -> DataType:

    # get data for the model
    data = pd.read_csv(f"{getcwd()}/insurance.csv")
    filtered_data = data[~data["smoker"].str.contains("no")]

    # split train and test data
    data_half_length = len(filtered_data.index) // 2

    train_data = filtered_data[:data_half_length]
    test_data = filtered_data[data_half_length:]

    # get train x and y values
    bmi_train = train_data[["bmi", "age"]].values * 1e-2
    charges_train = train_data[["charges"]].values * 1e-5

    # get the test x values
    bmi_test = test_data[["bmi", "age"]].values * 1e-2
    charges_test = test_data[["charges"]].values * 1e-5

    return {
        "bmi_train": bmi_train,
        "charges_train": charges_train,
        "bmi_test": bmi_test,
        "charges_test": charges_test
    }


def get_data_new() -> DataType:
    """Applies normalization to the data"""

    # get data for the model
    data = pd.read_csv(f"{getcwd()}/insurance.csv")
    filtered_data = data[~data["smoker"].str.contains("no")]
    selected = filtered_data[["bmi", "age", "charges"]]
    min_bmi = selected["bmi"].min()
    min_age = selected["bmi"].min()
    min_charges = selected["bmi"].min()
    max_bmi = selected["bmi"].max()
    max_age = selected["bmi"].max()
    max_charges = selected["bmi"].max()

    normalized = (selected-selected.min())/(selected.max()-selected.min())

    # split train and test data
    data_half_length = len(normalized.index) // 2

    train_data = normalized[:data_half_length]
    test_data = normalized[data_half_length:]

    # get train x and y values
    bmi_train = train_data[["bmi", "age"]].values
    charges_train = train_data[["charges"]].values

    # get the test x values
    bmi_test = test_data[["bmi", "age"]].values
    charges_test = test_data[["charges"]].values

    return {
        "bmi_train": bmi_train,
        "charges_train": charges_train,
        "bmi_test": bmi_test,
        "charges_test": charges_test,
        "min_bmi": selected["bmi"].min(),
        "min_age": selected["age"].min(),
        "min_charges": selected["charges"].min(),
        "max_bmi": selected["bmi"].max(),
        "max_age": selected["age"].max(),
        "max_charges": selected["charges"].max(),
    }


def make_predictions(
    model: LinearRegressionGD,
    data: DataType,
    normalization_fact: float = 1e5,
) -> StatsType:
    """Makes predictions with given a model and data"""

    # trains the model
    predicted_values = model.predict(data["bmi"])

    # get statistics
    mse = LinearRegressionGD.get_mean_squared_error(
        predicted_values=predicted_values * normalization_fact,
        real_values=data["charges"] * normalization_fact,
    )
    r2 = LinearRegressionGD.get_r2(
        predicted_values=predicted_values * normalization_fact,
        real_values=data["charges"] * normalization_fact,
    )

    return {
        "mse": round(mse, 4),
        "r2": round(r2, 4),
        "predicted": predicted_values,
    }


def train_model(
    data: DataType,
    iterations: int,
    normalization_fact: float = 1e5,
) -> Tuple[float, StatsType, LinearRegressionGD]:
    """Trains the model with given data and returns training time in seconds."""

    # train the model
    start_train_time = time.time()
    model = LinearRegressionGD(iterations=iterations, learning_rate=0.03)
    model.train(data["bmi_train"], data["charges_train"])
    finish_train_time = time.time()

    stats = make_predictions(
        model, {"bmi": data["bmi_train"], "charges": data["charges_train"]},
        normalization_fact=normalization_fact,
    )

    return round(finish_train_time - start_train_time, 4), stats, model


def main() -> None:
    """Main function of the measure experiments"""

    TRAINING_ITERATIONS = 100

    # Working with normalization
    data_new = get_data_new()

    new_iterations_list = []
    new_mse_list = []
    new_r2_list = []
    for i in range(TRAINING_ITERATIONS):
        iterations = (i + 1) * 200
        [_, stats, _] = train_model(
            data=data_new,
            iterations=iterations,
            normalization_fact=1,
        )
        new_iterations_list.append(iterations)
        new_mse_list.append(stats["mse"])
        new_r2_list.append(stats["r2"])

    # Working without normalization
    data_old = get_data_old()

    old_iterations_list = []
    old_mse_list = []
    old_r2_list = []
    for i in range(TRAINING_ITERATIONS):
        iterations = (i + 1) * 200
        [_, stats, _] = train_model(data=data_old, iterations=iterations)
        old_iterations_list.append(iterations)
        old_mse_list.append(stats["mse"])
        old_r2_list.append(stats["r2"])

    # Plotting
    _, ax = plt.subplots(2, 2)
    ax[0, 0].plot(old_iterations_list, old_mse_list, "tab:orange")
    ax[0, 0].set_title('MSE by Iterations')
    ax[0, 1].plot(old_iterations_list, old_r2_list, "tab:orange")
    ax[0, 1].set_title('R2 by Iterations')

    ax[1, 0].plot(new_iterations_list, new_mse_list, "tab:green")
    ax[1, 0].set_title('MSE by Iterations')
    ax[1, 1].plot(new_iterations_list, new_r2_list, "tab:green")
    ax[1, 1].set_title('R2 by Iterations')

    plt.tight_layout()
    plt.savefig("figures.png")


if __name__ == "__main__":
    main()
