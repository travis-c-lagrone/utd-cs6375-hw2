"""MultinomialNaiveBayes estimator and predictor for classification."""

from math import log
from typing import Any, Dict, List, Optional

from numpy import array, ndarray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class MultinomialNaiveBayes(BaseEstimator, ClassifierMixin):
    """A classifier that implements the multinomial naive bayes model.

    This class complies with the `Scikit-Learn protocols`_ for ``Estimator`` and ``Predictor``.

    .. _`Scikit-Learn protocols`: https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects

    """

    _dim: int
    _prior_probs: Dict[Optional[Any], float]
    _cond_probs: List[Dict[Optional[Any], float]]

    def fit(self, X: ndarray, y: ndarray) -> 'MultinomialNaiveBayes':
        """Estimate a multinomial naive bayes predicator over the given dataset.

        Args:
            X (ndarray): A 2d array of observations.
                Each row is an observation.
                Each column is a feature.
                Each entry is a nonnull cardinal number (e.g. absolute frequency or boolean indicator).
            y (ndarray): A 1d array of labels.
                The cardinality of ``y`` must be exactly equal to the number of observations in ``X`` (i.e. its "height").
                Each label is a nullable value of any equatable type, which is not necessarily homogenous.

        Returns:
            MultinomialNaiveBayes: This classifier, fit to the data as a multinomial naive bayes predictor.

        """
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

    def predict(self, X: ndarray) -> ndarray:
        """Classify samples using this multinomial naive bayes predictor.

        This ``MultinomialNaiveBayes`` instance must already be trained_.

        Args:
            X (ndarray): A 2d array of zero-or-more samples to classify.
                Each row is an observation.
                Each column is a feature.
                Each entry is a nonnull cardinal number (e.g. absolute frequency or boolean indicator).

        Returns:
            ndarray: The predicted labels as a 1d array.

        .. _trained: `multinomial_naive_bayes.MultinomialNaiveBayes.fit`

        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)

        preds: List[Optional[Any]] = []
        for row in X:  # type: ndarray
            scores = {}
            for c in self.classes_:
                score = log(self._prior_probs[c])
                for j in range(self._dim):
                    score += log(self._cond_probs[j][c]) * row[j]
                scores[c] = score
            pred = self.__argmax(scores)
            preds.append(pred)

        return array(preds)

    @staticmethod
    def __argmax(D: Dict[Optional[Any], float]) -> Optional[Any]:
        """Compute the key corresponding to the maximum value in the dictionary.

        Args:
            D (Dict[Optional[Any], float]): The dictionary over which to compute the argmax.
                The dictionary may not be empty.

        Returns:
            Optional[Any]: A maximum key from the set of keys corresponding to the maximum value(s).

        """
        max_val = max(D.values())
        max_key = max(k for k, v in D.items() if v == max_val)
        return max_key
