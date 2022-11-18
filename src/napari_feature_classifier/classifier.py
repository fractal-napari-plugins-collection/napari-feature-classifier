""" Classifier class file """
import os
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.napari_feature_classifier.data import ClassifierData
from src.napari_feature_classifier.utils import napari_info

CompatibleEstimators = Union[RandomForestClassifier, LogisticRegression]


def load_classifier(classifier_path):  # pylint: disable-msg=C0116
    with open(classifier_path, "rb") as f:  # pylint: disable-msg=C0103
        clf = pickle.loads(f.read())
    return clf


# pylint: disable-msg=C0116
def rename_classifier(classifier_path, new_name, delete_old_version=False):
    with open(classifier_path, "rb") as f:  # pylint: disable-msg=C0103
        clf = pickle.loads(f.read())
    clf.name = new_name
    clf.save()
    if delete_old_version:
        os.remove(classifier_path)


# pylint: disable-msg=R0902, R0913
class Classifier:
    """
    Classifier class to classify objects by a set of features

    Paramters
    ---------
    name: str
        Name of the classifier. E.g. "test". Will then be saved as test.clf
    features: pd.DataFrame
        Dataframe containing the features used for classification
    training_features: list
        List of features that are used for training the classifier
    method: str
        What classification method is used. Defaults to rfc => RandomForestClassifier
        Could also use "lrc" for a logistic regression classifier
    directory: pathlib.Path
        Directory where the classifier is saved
    index_columns: list or tuple
        Columns that are used to index the dataframe

    Attributes
    ----------
    name: str
        Name of the classifier. E.g. "test". Will then be saved as test.clf
    directory: pathlib.Path
        Directory where the classifier is saved
    clf: sklearn classifier class
        sklearn classifier that is used
    index_columns: list or tuple
        Columns that are used to index the dataframe
    train_data: pd.DataFrame
        Dataframe containing a "train" column to save annotations by the user
    predict_data: pd.DataFrame
        Dataframe containing a "predict" column to save predictions made by the
        classifier
    training_features: list
        List of features that are used for training the classifier
    data: pd.DataFrame
        Dataframe containing only yhe features define in "training_features"
    is_trained: boolean
        Flag of whether the classifier has been trained since data was added to it

    """

    def __init__(
            self,
            name: str,
            data: ClassifierData,
            training_features: list,
            clf: CompatibleEstimators = RandomForestClassifier(),
            directory=Path("."),
    ):
        self.name = name
        self.data = data
        self.training_features = training_features
        self.clf = clf
        self.directory = directory
        # TODO: Improve this flag. Currently user needs to set the flag to
        # false when changing training data. How can I automatically change
        # the flag whenever someone modifies self.train_data?
        # Could try something like this, but worried about the overhead:
        # https://stackoverflow.com/questions/6190468/how-to-trigger-function-on-value-change

        # Flag of whether the classifier has been trained since features have
        # changed last (new site added or train_data modified)
        self.is_trained = False
        # TODO: Check if data is numeric.
        # 1. Throw some exception for strings
        # 2. Handle nans: Inform the user.
        #   Some heuristic: If only < 10% of objects contain nan, ignore those objects
        #   If a feature is mostly nans (> 10%), ignore the feature (if multiple
        #   features are available) or show a warning
        #   Give the user an option to turn this off? E.g. via channel properties
        #   on the label image?
        #   => Current implementation should just give NaN results for all cells
        #  containing NaNs
        #   Have a way to notify the user of which features were NaNs? e.g. if
        #   one feature is always NaN, the classifier wouldn't do anything anymore
        # 3. Handle booleans: Convert to numeric 0 & 1.

    def add_data(self, data: pd.DataFrame):
        self.data.append_data_frame(data)
        self.is_trained = False

    @staticmethod
    def get_non_na_indices(df, message=""):  # pylint: disable-msg=C0103
        nan_values = df.isna()
        non_nan_indices = nan_values.sum(axis=1) == 0
        if nan_values.sum().sum() > 0:
            # Inform user about cells being removed and what features contain NaNs
            na_features = nan_values.sum()
            features_with_na = na_features[na_features > 0]
            napari_info(
                f"{(~non_nan_indices).sum()} cells were discarded during "
                f"{message} because they contain NaNs"
            )
            napari_info(
                f"The most NaNs were in {features_with_na.idxmax()} feature. "
                f"It contains {features_with_na.max()} NaNs"
            )
            if len(features_with_na) > 1:
                other_features = list(features_with_na.index)
                other_features.remove(features_with_na.idxmax())
                napari_info(
                    f"{len(features_with_na) - 1} other features also "
                    f"contained NaNs. Those are: {other_features}"
                )

        return non_nan_indices

    def train(self, ignore_nans=True):
        self.is_trained = True
        training_data = self.data.training_data
        training_annotations = self.data.training_annotations

        test_data = self.data.test_data
        test_annotations = self.data.test_annotations

        if ignore_nans:
            non_nan_training_indices = self.get_non_na_indices(training_data, message="training")
            non_nan_test_indices = self.get_non_na_indices(test_data, message="test")

            training_data = training_data.loc[non_nan_training_indices.values, :]
            training_annotations = training_annotations.loc[non_nan_training_indices.values, :]
            test_data = test_data.loc[non_nan_test_indices.values, :]
            test_annotations = test_annotations.loc[non_nan_test_indices.values, :]

        self.clf.fit(training_data, training_annotations.astype(int))

        # pylint: disable-msg=C0103
        f1 = f1_score(test_annotations.astype(int), self.clf.predict(test_data), average="macro")
        # napari_info("F1 score on test set: {}".format(f1))
        napari_info(
            f"F1 score on test set: {f1} \n"
            f"Annotations split into {len(training_data)} training and {len(test_data)} "
            "test samples. \n"
            f"Training set contains {self.get_count_per_class(training_annotations)}. \n"
            f"Test set contains {self.get_count_per_class(test_annotations)}."
        )
        self.data.predictions.loc[:] = self.predict(
            self.data.features, ignore_nans=ignore_nans
        ).reshape(-1, 1)
        return f1

    @staticmethod
    def get_count_per_class(train_col, background_class=0):
        """
        Generates a formated string for the number of samples per class, ignoring .

        Parameters
        ----------
        train_col: pandas.core.series.Series
            The pandas column containing the training data
        background_class:
            index of the background class that is not included in the count

        Returns
        -------
        str
            Formatted string containing the counts per class
        """
        output_str = ""
        classes = sorted(train_col.unique())
        counts = train_col.value_counts()
        for curr_class in classes:
            if curr_class != background_class:
                output_str += f"{counts[curr_class]} annotations for class {curr_class}, "
        return output_str[:-2]

    def predict(self, data, ignore_nans=True):
        if not self.is_trained:
            self.train()
        if ignore_nans:  # pylint: disable-msg=R1705
            # Does not throw an exception if data contains a NaN
            # Just returns NaN as a result for any cell containing NaNs
            non_nan = self.get_non_na_indices(
                data.loc[:, self.data.training_feature_names], message="prediction"
            )
            data.loc[:, "predict"] = np.nan
            data.loc[non_nan, "predict"] = self.clf.predict(
                data.loc[non_nan, self.data.training_feature_names]
            )
            return np.array(data["predict"])
        else:
            return self.clf.predict(data.loc[:, self.data.training_feature_names])

    def export_results(self, export_path):
        # Run the training & predictions on the current data if any new data
        # was added or training was modified
        if not self.is_trained:
            self.train()
        # Merge prediction data and training data
        output_df = pd.merge(self.data.predictions, self.data.annotations, on=self.data.index_columns)
        output_df.to_csv(export_path)

    def feature_importance(self):
        return OrderedDict(
            sorted(
                {
                    f: i
                    for f, i in zip(
                    self.training_features, self.clf.feature_importances_
                )
                }.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def most_important(self, n=5):  # pylint: disable-msg=C0103
        return list(self.feature_importance().keys())[:n]

    def save(self, new_name=None, directory=None):
        """
        Saves and optionally renames the classifier

        Parameters
        ----------
        new_name: str
            New name of the classifier. With or without .clf ending
        directory: pathlib.Path
            Path where the classifier will be saved. Optional, defaults to
            saving in the directory that is set in the self.directory variable,
            which itself defaults to the working directory
        """
        if new_name is not None:
            if new_name.endswith(".clf"):
                new_name = new_name[:-4]
            self.name = new_name
        pickle_dump = pickle.dumps(self)
        if directory is not None:
            self.directory = directory
        try:
            # pylint: disable-msg=C0103
            with open(self.directory / (self.name + ".clf"), "wb") as f:
                f.write(pickle_dump)
        # Handle edge case with old classifiers that didn't have a directory
        # attribute
        except AttributeError:
            if directory is not None:
                self.directory = directory
            else:
                self.directory = Path(".")
            # pylint: disable-msg=C0103
            with open(self.directory / (self.name + ".clf"), "wb") as f:
                f.write(pickle_dump)
