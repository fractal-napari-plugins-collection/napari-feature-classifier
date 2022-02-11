from collections import OrderedDict
from zlib import crc32
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os
import warnings
from .utils import napari_info


def make_identifier(df):
    str_id = df.apply(lambda x: "_".join(map(str, x)), axis=1)
    return str_id


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(hash(identifier))) & 0xFFFFFFFF < test_ratio * 2**32


def load_classifier(classifier_path):
    with open(classifier_path, "rb") as f:
        clf = pickle.loads(f.read())
    return clf


def rename_classifier(classifier_path, new_name, delete_old_version=False):
    with open(classifier_path, "rb") as f:
        clf = pickle.loads(f.read())
    clf.name = new_name
    clf.save()
    if delete_old_version:
        os.remove(classifier_path)


class Classifier:
    def __init__(
        self, name, features, training_features, method="rfc", index_columns=None
    ):
        # TODO: Think about chainging the not classified class to NaN instead of 0
        # (when manually using the classifier, a user may provide 0s as training input when predicting some binary result)
        self.name = name
        if method == "rfc":
            self.clf = RandomForestClassifier()
        elif method == "lrc":
            self.clf = LogisticRegression()
        full_data = features
        full_data.loc[:, "train"] = 0
        full_data.loc[:, "predict"] = 0
        self.index_columns = index_columns
        self.train_data = full_data[["train"]]
        self.predict_data = full_data[["predict"]]
        self.training_features = training_features
        self.data = full_data[self.training_features]
        # TODO: Improve this flag. Currently user needs to set the flag to false when changing training data. How can I automatically change the flag whenever someone modifies self.train_data?
        # Could try something like this, but worried about the overhead: https://stackoverflow.com/questions/6190468/how-to-trigger-function-on-value-change
        self.is_trained = False  # Flag of whether the classifier has been trained since features have changed last (new site added or train_data modified)
        # TODO: Check if data is numeric.
        # 1. Throw some exception for strings
        # 2. Handle nans: Inform the user.
        #   Some heuristic: If only < 10% of objects contain nan, ignore those objects
        #   If a feature is mostly nans (> 10%), ignore the feature (if multiple features are available) or show a warning
        #   Give the user an option to turn this off? E.g. via channel properties on the label image?
        #   => Current implementation should just give NaN results for all cells containing NaNs
        #   Have a way to notify the user of which features were NaNs? e.g. if one feature is always NaN, the classifier wouldn't do anything anymore
        # 3. Handle booleans: Convert to numeric 0 & 1.

    @staticmethod
    def train_test_split(df, test_perc=0.2, index_columns=None):
        in_test_set = make_identifier(df.reset_index()[list(index_columns)]).apply(
            test_set_check, args=(test_perc,)
        )

        if in_test_set.sum() == 0:
            warnings.warn(
                "Not enough training data. No training data was put in the test set and classifier will fail."
            )
        if in_test_set.sum() == len(in_test_set):
            warnings.warn(
                "Not enough training data. All your selections became test data and there is nothing to train the classifier on"
            )
        return df.iloc[~in_test_set.values, :], df.iloc[in_test_set.values, :]

    def add_data(self, features, training_features, index_columns):
        # Check that training features agree with already existing training features
        assert training_features == self.training_features, (
            "The training "
            "features provided to the classifier are different to what has "
            "been used for training so far. This has not been implemented "
            "yet. Old vs. new: {} vs. {}".format(
                self.training_features, training_features
            )
        )

        # Check if data with the same index already exists. If so, do nothing
        assert index_columns == self.index_columns, (
            "The newly added dataframe "
            "uses different index columns "
            "than what was used in the "
            "classifier before: New {}, "
            "before {}".format(index_columns, self.index_columns)
        )
        # Check which indices already exist in the data, only add the others
        new_indices = self._index_not_in_other_df(features, self.train_data)
        new_data = features.loc[new_indices["index_new"]]
        if len(new_data.index) == 0:
            # No new data to be added: The classifier is being loaded for a
            # site where the data has been loaded before
            # TODO: Is there a low-priority logging this could be sent to?
            # Not a warning, just info or debug
            pass
        else:
            new_data["train"] = 0
            new_data["predict"] = 0
            self.train_data = self.train_data.append(new_data[["train"]])
            self.predict_data = self.predict_data.append(new_data[["predict"]])
            self.data = self.data.append(new_data[training_features])
        self.is_trained = False

    @staticmethod
    def _index_not_in_other_df(df1, df2):
        # Function checks which indices of df1 already exist in the indices of df2.
        # Returns a boolean pd.DataFrame with a 'index_preexists' column
        df_overlap = pd.DataFrame(index=df1.index)
        for df1_index in df1.index:
            if df1_index in df2.index:
                df_overlap.loc[df1_index, "index_new"] = False
            else:
                df_overlap.loc[df1_index, "index_new"] = True
        return df_overlap

    @staticmethod
    def get_non_na_indices(df, message=""):
        nan_values = df.isna()
        non_nan_indices = nan_values.sum(axis=1) == 0
        if nan_values.sum().sum() > 0:
            # Inform user about cells being removed and what features contain NaNs
            na_features = nan_values.sum()
            features_with_na = na_features[na_features > 0]
            napari_info(
                "{} cells were discarded during {} because they contain NaNs".format(
                    (~non_nan_indices).sum(), message
                )
            )
            napari_info(
                "The most NaNs were in {} feature. It contains {} NaNs".format(
                    features_with_na.idxmax(), features_with_na.max()
                )
            )
            if len(features_with_na) > 1:
                other_features = list(features_with_na.index)
                other_features.remove(features_with_na.idxmax())
                napari_info(
                    "{} other features also contained NaNs. Those are: {}".format(
                        len(features_with_na) - 1, other_features
                    )
                )

            return non_nan_indices
        else:
            return non_nan_indices

    def train(self, ignore_nans=True):
        # TODO: Select training data differently. 0 could be a valid training input
        # Load only training features. The data df will also contain the prior predictions, which can lead to issues
        self.is_trained = True
        training_data = self.data.loc[
            self.train_data["train"] > 0, self.training_features
        ]
        training_results = self.train_data[self.train_data["train"] > 0]

        if ignore_nans:
            non_nan_indices = self.get_non_na_indices(training_data, message="training")
            X_train, X_test = self.train_test_split(
                training_data[non_nan_indices], index_columns=self.index_columns
            )
            y_train, y_test = self.train_test_split(
                training_results[non_nan_indices], index_columns=self.index_columns
            )
        else:
            X_train, X_test = self.train_test_split(
                training_data, index_columns=self.index_columns
            )
            y_train, y_test = self.train_test_split(
                training_results, index_columns=self.index_columns
            )

        assert np.all(X_train.index == y_train.index)
        assert np.all(X_test.index == y_test.index)

        self.clf.fit(X_train, y_train["train"])

        f1 = f1_score(y_test, self.clf.predict(X_test), average="macro")
        #napari_info("F1 score on test set: {}".format(f1))
        napari_info(
            "F1 score on test set: {} \n"
            "Annotations split into {} training and {} test samples. \n"
            "Training set contains {}. \n Test set contains {}.".format(
                f1,
                len(X_train),
                len(X_test),
                self.get_count_per_class(y_train['train']),
                self.get_count_per_class(y_test['train']),
            )
        )
        self.predict_data.loc[:] = self.predict(
            self.data, ignore_nans=ignore_nans
        ).reshape(-1, 1)
        return f1

    @staticmethod
    def get_count_per_class(train_col, background_class = 0):
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
        output_str = ''
        classes = sorted(train_col.unique())
        counts = train_col.value_counts()
        for cl in classes:
            if cl != background_class:
                output_str += '{} objects for class {}, '.format(counts[cl], cl)
        return output_str[:-2]

    def predict(self, data, ignore_nans=True):
        if not self.is_trained:
            self.train()
        if ignore_nans:
            # Does not throw an exception if data contains a NaN
            # Just returns NaN as a result for any cell containing NaNs
            non_nan = self.get_non_na_indices(
                data.loc[:, self.training_features], message="prediction"
            )
            data.loc[:, "predict"] = np.nan
            data.loc[non_nan, "predict"] = self.clf.predict(
                data.loc[non_nan, self.training_features]
            )
            return np.array(data["predict"])
        else:
            return self.clf.predict(data.loc[:, self.training_features])

    def export_results_single_site(self, export_path):
        # Run the training & predictions on the current data if any new data was added or training was modified
        if not self.is_trained:
            self.train()
        # Merge prediction data and training data
        output_df = pd.merge(self.predict_data, self.train_data, on=self.index_columns)
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

    def most_important(self, n=5):
        return list(self.feature_importance().keys())[:n]

    def save(self, new_name=None):
        if new_name is not None:
            self.name = new_name
        s = pickle.dumps(self)
        with open(self.name + ".clf", "wb") as f:
            f.write(s)
