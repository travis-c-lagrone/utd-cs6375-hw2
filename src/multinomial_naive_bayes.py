from math import log
from typing import Any, Dict, List, Optional

from numpy import array, ndarray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class MultinomialNaiveBayes(BaseEstimator, ClassifierMixin):  # noqa: D101

    _dim: int
    _prior_probs: Dict[Optional[Any], float]
    _cond_probs: List[Dict[Optional[Any], float]]

    def fit(self, X: ndarray, y: ndarray) -> 'MultinomialNaiveBayes':  # noqa: D102
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y
        self.classes_ = unique_labels(y)

        self._dim = d = X.shape[1]
        n = len(y)

        self._prior_probs = {}
        self._cond_probs = [{} for _ in range(d)]

        for c in self.classes_:
            self._prior_probs[c] = sum(map(lambda c_y: c_y == c, y)) / n
            d_sums = X.sum(axis=0, where=X[y == c, :])
            denom = d_sums.sum() + d
            for j in range(d):
                self._cond_probs[j][c] = (d_sums[j] + 1) / denom

        return self

    def predict(self, X: ndarray) -> ndarray:  # noqa: D102
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)

        preds: List[Optional[Any]] = []
        for row in X:  # type: ndarray
            scores = {}
            for c in self.classes_:
                score = log(self._prior_probs[c])
                for j in range(self._dim):
                    score += log(self._cond_probs[j][c])
                scores[c] = score
            pred = self._argmax(scores)
            preds.append(pred)

        return array(preds)

    @staticmethod
    def _argmax(D: Dict[Optional[Any], float]) -> Optional[Any]:
        return max(D.items(), key=lambda kv: kv[1])[1]
