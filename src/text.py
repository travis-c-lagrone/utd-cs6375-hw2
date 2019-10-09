from collections import Counter
from typing import Collection, Dict, List, Set

from nltk import word_tokenize
from numpy import array, ndarray


class BagOfWords:
    """A textual data preprocessor that implements the bag-of-words model.

    This class complies with the Scikit-Learn `Transformer protocol`_.

    .. _`Transformer protocol`: https://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects

    """

    def transform(self, docs: Collection[str]) -> ndarray:
        """Encode a collection of textual documents with the bag-of-words model.

        Args:
            docs (Collection[str]): Zero-or-more nonnull text documents.
                Each element is the complete textual _content_ of a document.

        Returns:
            List[str]: The lexicographically-ordered vocabulary extracted from the collection of documents.
                The cardinality of the vocabulary is exactly the width of the returned dataset.
                The ordering of the vocabulary is exactly the ordering of the corresponding columns in the returned dataset.
                The exact nature of the tokenization algorithm is an opaque implementation detail.

            ndarray: The 2d integral array representing the bag-of-words of the documents.
                Each row is a document.
                Each column is a word.
                Each entry is the absolute frequency the corresponding word in the corresponding document.

        """
        words: Set[str] = set()
        bags: List[Dict[str, int]] = list()

        for doc in docs:
            tokens = word_tokenize(doc)
            bag = Counter(tokens)

            words.update(bag.keys())
            bags.append(bag)

        ordered_words = sorted(words)
        standardized_bags = [[bag[word] for word in words] for bag in bags]

        return ordered_words, array(standardized_bags)


class Bernoulli:  # noqa: D101

    def transform(self, docs: Collection[str]) -> ndarray:
        pass  # TODO Bernoulli.transform(self, docs)
