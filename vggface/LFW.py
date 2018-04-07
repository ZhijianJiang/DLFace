#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of loading lfw data

Reference: http://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py
"""

import ssl
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context


class LFWLoader:
    """

    Params:
        data_home: path to store LFW data
        target_names (ndarray): list of target names
    """

    def __init__(self, data_home='~/Downloads/scikit_learn_data'):
        """
        Args:
             data_home (str, optional): Default to '~/Downloads/scikit_learn_data'
        """
        self.data_home = data_home

    def load_lfw_data(self,
                      funneled=True,
                      resize=1,
                      min_faces_per_person=2,
                      test_size=0.25,
                      random_state=42):
        """
        Args:
            funneled (boolean, optional): Default to True. Download the funneled dataset.
            resize (float, optional): Default to 1. Ratio to resize the face pictures.
            min_faces_per_person (int, optional): Default to 2. The extracted dataset will only retain pictures of people that have at least min_faces_per_person different pictures.
            test_size (float, int, optional): Default to 0.25. If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.
            random_state (int, RandomState instance, optional): Default to 42. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator.

        Returns:
            X_train (ndarray)
            X_test (ndarray)
            y_train (ndarray): onehot label
            y_test (ndarray): onehot label
        """
        # #############################################################################
        # Download the data, if not already on disk and load it as numpy arrays

        lfw_people = fetch_lfw_people(
            data_home=self.data_home,
            funneled=funneled,
            resize=resize,
            min_faces_per_person=min_faces_per_person,
            color=True
            )

        # introspect the images arrays to find the shapes (for plotting)
        n_samples, d, h, w = lfw_people.images.shape

        # for machine learning we use the 2 data directly (as relative pixel
        # positions info is ignored by this model)
        X = lfw_people.images
        n_features = X.shape[1]

        # the label to predict is the id of the person
        y = lfw_people.target
        y = OneHotEncoder(sparse=False).fit_transform(np.reshape(y, (len(y), 1)))

        self.target_names = lfw_people.target_names # store the list of target names
        n_classes = self.target_names.shape[0]

        print("Total dataset size:")
        print("n_samples: %d" % n_samples)
        print("n_features: %d" % n_features)
        print("n_classes: %d" % n_classes)
        # print("image size: " + str(h) + ", " + str(w))

        # #############################################################################
        # Split into a training set and a test set using a stratified k fold

        # split into a training and testing set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        print('Done!')
        return X_train, X_test, y_train, y_test

def main():
    """
    For test
    """
    X_train, X_test, y_train, y_test = LFWLoader().load_lfw_data()


if __name__ == "__main__":
    main()

