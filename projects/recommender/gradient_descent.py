# Created by: Topiltzin Hernández Mares
# Created at: 07/10/2021
# GitHub: https://github.com/Topi99
import itertools
from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix


class GradientDescentMF:
    """Matrix Factorization using Gradient Descent as minimizer function"""

    def __init__(
        self,
        item_user: csr_matrix,
        features: int = 1,
        verbose: bool = False,
        learning_rate: float = 0.1,
        iterations: int = 1000,
    ) -> None:
        self._verbose = verbose
        self._learning_rate = learning_rate
        self._iterations = iterations

        self._item_user = item_user.copy().toarray()
        self._log(f"{self._item_user.shape = }")
        self._features = features
        self._users_count: int = self._item_user.shape[0]
        self._items_count: int = self._item_user.shape[1]
        self._log(f"{self._users_count = }")
        self._log(f"{self._items_count = }")

        self._user_features = np.random.uniform(
            low=0.1, high=0.9, size=(self._users_count, self._features),
        )
        self._features_item = np.random.uniform(
            low=0.1, high=0.9, size=(self._features, self._items_count),
        )

    def mean_squared_error(self) -> np.ndarray:
        """
        Mean Squared Error function for comparing dot product of user-feature
        row and feature-item column to user user-item cell.
        :return: ndarray with the mean squared error.
        """

        matrix_product = np.matmul(self._user_features, self._features_item)
        return np.sum((self._item_user - matrix_product) ** 2)

    def _single_gradient_user(
        self, user_row: int, item_col: int, feature_index: int,
    ) -> float:
        """
        Computes gradient of a single user-item cell to a single user-feature
        cell.
        :param user_row:
        :param item_col:
        :param feature_index:
        :return: the new value for the feature using gradient function
        """

        user_row_feature = self._user_features[user_row, :]
        item_col_feature = self._features_item[:, item_col]
        user_item_rating = float(self._item_user[user_row, item_col])
        prediction = float(np.dot(user_row_feature, item_col_feature))

        feature_item = float(item_col_feature[feature_index])

        # return the new value using the gradient function
        return 2 * (user_item_rating - prediction) * feature_item

    def _single_gradient_item(
        self, user_row: int, item_col: int, feature_index: int,
    ) -> float:
        """
        Computes gradient of a single user-item cell to a single feature-item
        cell.
        :param user_row:
        :param item_col:
        :param feature_index:
        :return: the new value for the feature using gradient function
        """

        user_row_feature = self._user_features[user_row, :]
        item_col_feature = self._features_item[:, item_col]
        user_item_rating = float(self._item_user[user_row, item_col])
        prediction = float(np.dot(user_row_feature, item_col_feature))

        feature_item = float(user_row_feature[feature_index])

        # return the new value using the gradient function
        return 2 * (user_item_rating - prediction) * feature_item

    def _user_feature_gradient(
        self, user_row: int, user_feature_index: int,
    ) -> float:
        """
        Averages the gradients of a single user-item row with respect to a
        single user-feature parameter
        :param user_row:
        :param user_feature_index:
        :return:
        """

        gradients_acum = 0
        for col in range(0, self._items_count):
            gradients_acum += self._single_gradient_user(
                user_row=user_row,
                item_col=col,
                feature_index=user_feature_index,
            )
        return gradients_acum / self._users_count

    def _item_feature_gradient(
        self, item_col: int, feature_item_index: int,
    ) -> float:
        """
        Averages the gradients of a single user-item column with respect to a
        single feature-item parameter
        :param item_col:
        :param feature_item_index:
        :return:
        """

        gradients_acum = 0
        for row in range(0, self._users_count):
            gradients_acum += self._single_gradient_user(
                user_row=row,
                item_col=item_col,
                feature_index=feature_item_index,
            )
        return gradients_acum / self._users_count

    def _update_user_features(self) -> None:
        """Updates every user-feature parameter"""

        for user_row in range(0, self._users_count):
            for feature_col in range(0, self._features):
                self._user_features[user_row, feature_col] += (
                    self._learning_rate * self._user_feature_gradient(
                        user_row=user_row, user_feature_index=feature_col,
                    )
                )

    def _update_item_features(self) -> None:
        """Updates every feature-item parameter"""

        for feature_row in range(0, self._features):
            for item_col in range(0, self._items_count):
                self._features_item[feature_row, item_col] += (
                    self._learning_rate * self._item_feature_gradient(
                        item_col=item_col, feature_item_index=feature_row,
                    )
                )

    def train(self) -> None:
        """Trains the current model"""

        for i in range(self._iterations):
            self._update_user_features()
            self._update_item_features()

            if self._verbose and i % 50 == 0:
                print(f"Mean Squared Error in iteration #{i}")
                print(f"\t{self.mean_squared_error():.4f}")

        if self._verbose:
            trained_model = np.dot(self._user_features, self._features_item)
            print(f"{trained_model = }")
            print(f"{self._item_user = }")

    def recommend(
        self,
        user_id: int,
        user_items: csr_matrix,
        n: int = 3,
    ) -> List[Tuple[int, float]]:
        user_feature = self._user_features[user_id]
        self._log(f"{user_feature = }")

        liked = set()
        liked.update(user_items[user_id].indices)
        self._log(f"{liked = }")

        # calculate top N items, removing the users own liked items
        # scores = self._features_item.dot(user_feature)
        scores = np.dot(user_feature, self._features_item)
        self._log(f"{scores = }")

        count = n + len(liked)
        if count < len(scores):
            ids = np.argpartition(scores, -count)[-count:]
            best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])
            self._log(f"{best = }")
        else:
            best = sorted(enumerate(scores), key=lambda x: -x[1])
            self._log(f"{best = }")

        return list(itertools.islice(
            (rec for rec in best if rec[0] not in liked),
            n)
        )

    def _log(self, message: str) -> None:
        if self._verbose:
            print(message)
