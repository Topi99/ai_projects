import pickle
from typing import Optional

import pandas as pd
from scipy import sparse


class DataHandler:
    """Data handler class for the recommender experiments."""

    str2num = {
        "NOVICE": 1, "INTERMEDIATE": 2, "ADVANCED": 3, "EXPERT": 4
    }

    def __init__(self, file_name: str):
        self._file_name = file_name
        self._data: Optional[pd.DataFrame] = None
        self._sparse_item_user: Optional[sparse.csr_matrix] = None
        self._sparse_user_item: Optional[sparse.csr_matrix] = None

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = self._read_data()
        return self._data

    def _read_data(self) -> pd.DataFrame:
        # read the raw data
        raw_data = pd.read_csv(self._file_name)

        data = raw_data.dropna()
        del raw_data
        data = data.copy()

        # transform levels to numbers
        data["rating"] = data["level"].apply(
            lambda x: self.str2num.get(x, None)
        )

        # transform user UUIDs to categorical values
        data["uuid"] = data["uuid"].astype("category")

        # transform skill names to categorical values
        data["skill_name"] = data["skill_name"].astype("category")
        # normalize user_ids as categorical codes
        data["user_id"] = data["uuid"].cat.codes

        # normalize skill_ids as categorical codes
        data["skill_id"] = data["skill_name"].cat.codes

        return data

    @property
    def sparse_item_user(self) -> sparse.csr_matrix:
        """
        create sparse matrix item-user this is going to be used for training
        the model.

        :return: the sparse item-user csr_matrix
        """
        if not self._sparse_item_user:
            self._sparse_item_user = sparse.csr_matrix(
                (
                    self.data["rating"].astype(float),
                    (self.data["skill_id"], self.data["user_id"]),
                )
            )
        print(f"{self._sparse_item_user.shape = }")
        return self._sparse_item_user

    @property
    def sparse_user_item(self) -> sparse.csr_matrix:
        """
        create sparse matrix user-item this is going to be used for making
        recommendations.

        :return: the sparse user-item csr_matrix
        """
        if not self._sparse_user_item:
            self._sparse_user_item = sparse.csr_matrix(
                (
                    self.data["rating"].astype(float),
                    (self.data["user_id"], self.data["skill_id"]),
                )
            )
        print(f"{self._sparse_user_item.shape = }")
        return self._sparse_user_item

    def export_data(self) -> None:
        users_asc = self.data.uuid.cat.categories
        skills_asc = self.data.skill_name.cat.categories
        user2code = {user: code for code, user in enumerate(users_asc)}
        skill2code = {skill: code for code, skill in enumerate(skills_asc)}
        code2user = {code: user for code, user in enumerate(users_asc)}
        code2skill = {code: skill for code, skill in enumerate(skills_asc)}

        export_data = {
            "users_asc": users_asc,
            "skills_asc": skills_asc,
            "user2code": user2code,
            "skill2code": skill2code,
            "code2skill": code2skill,
            "code2user": code2user,
            "model": self,
        }

        # export model with pickle
        with open(f"model_{self._file_name}.pickle", "wb") as handle:
            pickle.dump(
                export_data, handle, protocol=pickle.HIGHEST_PROTOCOL,
            )
