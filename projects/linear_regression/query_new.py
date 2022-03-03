import pickle
from typing import Optional

import numpy as np

from projects.linear_regression.gradient_descent import LinearRegressionGD

if __name__ == "__main__":
    with open("model_new.pickle", "rb") as handle:
        dump = pickle.load(handle)
        model: Optional[LinearRegressionGD] = dump.get("model", None)
        min_bmi: float = dump.get("min_bmi", 0.)
        min_age: float = dump.get("min_age", 0.)
        min_charges: float = dump.get("min_charges", 0.)
        max_bmi: float = dump.get("max_bmi", 0.)
        max_age: float = dump.get("max_age", 0.)
        max_charges: float = dump.get("max_charges", 0.)

    if not model:
        raise Exception("No model exported yet! First train the model to query.")

    other = True
    while other:
        age = int(input("Give me your age: "))
        bmi = float(input("Give me your bmi: "))

        normalized_age = (age-min_age)/(max_age-min_age)
        normalized_bmi = (bmi-min_bmi)/(max_bmi-min_bmi)

        normalized_prediction_arr = model.predict(
            np.array([[normalized_bmi, normalized_age]]),
        )

        normalized_prediction = normalized_prediction_arr[0][0]
        prediction = normalized_prediction * (max_charges - min_charges) + min_charges
        print(f"Your predicted medical insurance charge: ${prediction:.2f}")

        resp = input("\nDo other query? [y/N]: ")
        other = resp == "Y" or resp == "y"
