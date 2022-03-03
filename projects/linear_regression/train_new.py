import pickle

from projects.linear_regression.measure import train_model, make_predictions, get_data_new

if __name__ == "__main__":
    # get data for the model
    data = get_data_new()

    # train the model
    [time_train, train_stats, model] = train_model(
        data=data, iterations=5000, normalization_fact=1,
    )

    print(f"Time to train: {time_train} seconds")
    print(f"Mean Squared Error in train: {train_stats['mse']:.2f}")
    print(f"Coefficient of determination in train: {train_stats['r2']:.2f}\n")

    # test the model
    test_stats = make_predictions(
        model,
        {"bmi": data["bmi_test"], "charges": data["charges_test"]},
        normalization_fact=1,
    )

    print(f"Mean Squared Error in test: {test_stats['mse']:.2f}")
    print(f"Coefficient of determination in test: {test_stats['r2']:.2f}\n")

    # export model
    with open("model_new.pickle", "wb") as handle:
        pickle.dump({
            "model": model,
            "min_bmi": data["min_bmi"],
            "min_age": data["min_age"],
            "min_charges": data["min_charges"],
            "max_bmi": data["max_bmi"],
            "max_age": data["max_age"],
            "max_charges": data["max_charges"],
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Model exported! Ready to query")
