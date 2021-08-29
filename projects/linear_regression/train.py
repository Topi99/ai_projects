import pickle
from os import getcwd

import pandas as pd

from projects.linear_regression.gradient_descent import LinearRegressionGD


if __name__ == "__main__":
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

    # train the model
    model = LinearRegressionGD()
    model.train(bmi_train, charges_train)

    # test the model
    predicted_values = model.predict(bmi_train)

    # get statistics
    mse_train = LinearRegressionGD.get_mean_squared_error(
        predicted_values=predicted_values * 1e5, real_values=charges_train * 1e5,
    )
    r2_train = LinearRegressionGD.get_r2(
        predicted_values=predicted_values * 1e5, real_values=charges_train * 1e5,
    )

    print(f"Mean Squared Error in train: {mse_train:.2f}")
    print(f"Coefficient of determination in train: {r2_train:.2f}\n")

    # test the model
    predicted_values = model.predict(bmi_test)
    print(predicted_values)
    # get statistics
    mse_test = LinearRegressionGD.get_mean_squared_error(
        predicted_values=predicted_values * 1e5, real_values=charges_test * 1e5,
    )
    r2_test = LinearRegressionGD.get_r2(
        predicted_values=predicted_values * 1e5, real_values=charges_test * 1e5,
    )

    print(f"Mean Squared Error in test: {mse_test:.2f}")
    print(f"Coefficient of determination in test: {r2_test:.2f}\n")

    # export model
    with open("model.pickle", "wb") as handle:
        pickle.dump({"model": model}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Model exported! Ready to query")
