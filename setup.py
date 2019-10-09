from setuptools import setup, find_packages

setup(
    name='hw2',
    version='0.1',
    packages=find_packages()
)

import nltk  # noqa: E402
nltk.download('punkt')
