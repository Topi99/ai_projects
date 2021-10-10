import pandas as pd
import scipy.sparse as sparse
import implicit.als as als

from projects.recommender.gradient_descent import GradientDescentMF

FILE_NAME = "skills_ordered.csv"

if __name__ == "__main__":
    # read the data
    raw_data = pd.read_csv(FILE_NAME)

    # drop empty rows just in case
    data = raw_data.dropna()
    del raw_data
    data = data.copy()

    # transform levels to numbers
    str2num = {"NOVICE": 1, "INTERMEDIATE": 2, "ADVANCED": 3, "EXPERT": 4}
    data["rating"] = data["level"].apply(lambda x: str2num.get(x, None))

    # transform user UUIDs to categorical values
    data["uuid"] = data["uuid"].astype("category")

    # transform skill names to categorical values
    data["skill_name"] = data["skill_name"].astype("category")

    # normalize user_ids as categorical codes
    data["user_id"] = data["uuid"].cat.codes
    print(f"{data['user_id'] = }")

    # normalize skill_ids as categorical codes
    data["skill_id"] = data["skill_name"].cat.codes

    # create sparse matrix item-user
    # this is going to be used for training the model
    sparse_item_user = sparse.csr_matrix(
        (
            data["rating"].astype(float),
            (data["skill_id"], data["user_id"]),
        )
    )

    # create sparse matrix user-item
    # this is going to be used for making recommendations
    sparse_user_item = sparse.csr_matrix(
        (
            data["rating"].astype(float),
            (data["user_id"], data["skill_id"]),
        )
    )

    # train the model
    # recommender = als.AlternatingLeastSquares(
    #     factors=20, regularization=0.1, iterations=20,
    # )
    # alpha = 40
    # data_confidence = (sparse_item_user * alpha).astype('double')
    # recommender.fit(data_confidence)
    recommender = GradientDescentMF(
        item_user=sparse_item_user, verbose=True, features=3, iterations=200
    )
    recommender.train()

    # making recommendations
    users_asc = data.uuid.cat.categories
    skills_asc = data.skill_name.cat.categories
    user2code = {user: code for code, user in enumerate(users_asc)}
    skill2code = {skill: code for code, skill in enumerate(skills_asc)}
    code2user = {code: user for code, user in enumerate(users_asc)}
    code2skill = {code: skill for code, skill in enumerate(skills_asc)}

    USER_UUID = "e0d0e4bd-af6c-4812-b463-d1f798cd3e74"
    user_id = user2code[USER_UUID]
    recommendations = recommender.recommend(user_id, sparse_user_item, n=4)
    print(
        "Recommendations: ",
        ", ".join([code2skill[idx] for idx, _ in recommendations])
    )
