import pandas as pd
import scipy.sparse as sparse
import implicit.als as als

from projects.recommender.data_handler import DataHandler
from projects.recommender.gradient_descent import GradientDescentMF

FILE_NAME = "skills_simple.csv"

if __name__ == "__main__":
    # read the data
    data_handler = DataHandler(file_name=FILE_NAME)
    data = data_handler.data
    sparse_user_item = data_handler.sparse_user_item

    # train the model
    recommender = GradientDescentMF(
        user_item=sparse_user_item.A.astype('double'),
        verbose=True,
        features=20,
        learning_rate=0.01,
        iterations=400,
    )
    recommender.train()

    # making recommendations
    users_asc = data.uuid.cat.categories
    skills_asc = data.skill_name.cat.categories
    user2code = {user: code for code, user in enumerate(users_asc)}
    skill2code = {skill: code for code, skill in enumerate(skills_asc)}
    code2user = {code: user for code, user in enumerate(users_asc)}
    code2skill = {code: skill for code, skill in enumerate(skills_asc)}

    other = 1
    while other:
        user_uuid = input("Enter your UUID: ")
        user_id = user2code[user_uuid]

        rankings = recommender.recommend(user_id, sparse_user_item, n=4)
        print(f"{rankings =}")
        print(
            "Recommendations: ",
            ", ".join([code2skill[idx] for idx, _ in rankings])
        )

        answer = input("Another recommendation? [y/n] ")
        other = 1 if answer == "y" else 0
