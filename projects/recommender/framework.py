import implicit.als as als

from projects.recommender.data_handler import DataHandler

FILE_NAME = "skills_simple.csv"

if __name__ == "__main__":
    # read the data
    data_handler = DataHandler(file_name=FILE_NAME)
    data = data_handler.data
    sparse_user_item = data_handler.sparse_user_item
    sparse_item_user = data_handler.sparse_item_user

    # train the model
    recommender = als.AlternatingLeastSquares(
        factors=20, regularization=0.05, iterations=500,
    )
    alpha = 40
    data_confidence = (sparse_item_user * alpha).astype('double')
    recommender.fit(sparse_item_user)

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

        rankings = recommender.recommend(user_id, sparse_user_item, N=4)
        print(f"{rankings =}")
        print(
            "Recommendations: ",
            ", ".join([code2skill[idx] for idx, _ in rankings])
        )

        answer = input("Another recommendation? [y/n] ")
        other = 1 if answer == "y" else 0
