import pickle

from projects.linear_regression.measure import get_data_old, train_model, make_predictions

if __name__ == "__main__":
    # get data for the model
    data = get_data_old()

    # train the model
    [time_train, train_stats, model] = train_model(data=data, iterations=12000)

    print(f"Time to train: {time_train} seconds")
    print(f"Mean Squared Error in train: {train_stats['mse']:.2f}")
    print(f"Coefficient of determination in train: {train_stats['r2']:.2f}\n")

    # test the model
    test_stats = make_predictions(
        model, {"bmi": data["bmi_test"], "charges": data["charges_test"]},
    )

    print(f"Mean Squared Error in test: {test_stats['mse']:.2f}")
    print(f"Coefficient of determination in test: {test_stats['r2']:.2f}\n")

    # export model
    with open("model_old.pickle", "wb") as handle:
        pickle.dump({"model": model}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Model exported! Ready to query")
