import pickle
from typing import Optional

import numpy as np

from projects.linear_regression.gradient_descent import LinearRegressionGD

if __name__ == "__main__":
    with open("model.pickle", "rb") as handle:
        dump = pickle.load(handle)
        model: Optional[LinearRegressionGD] = dump.get("model", None)

    if not model:
        raise Exception("No model exported yet! First train the model to query.")

    other = True
    while other:
        age = int(input("Give me your age: "))
        bmi = float(input("Give me your bmi: "))

        prediction_arr = model.predict(np.array([[bmi * 1e-2, age * 1e-2]])) * 1e5

        prediction = prediction_arr[0][0]
        print(f"Your predicted medical insurance charge: ${prediction:.2f}")

        resp = input("\nDo other query? [y/N]: ")
        other = resp == "Y" or resp == "y"
