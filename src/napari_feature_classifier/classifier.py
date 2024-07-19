"""Core classifier class and helper functions."""
import logging
import pickle
import random
import string
from typing import Sequence

import pandas as pd
import pandera as pa
import xxhash

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier


# TODO: define an interface for compatible classifiers (m.b. a subset of
# sklearn Estimators?)
class Classifier:
    """Classifier class for napari-feature-classifier.

    Paramters
    ---------
    feature_names: Sequence[str]
        The names of the features that are used for classification
    class_names: Sequence[str]
        The names of the classes. It's an ordered list that is matched to
        annotations [1, 2, 3, ...]
    classifier: sklearn classifier
        The classifier that is used for classification. Default is a
        RandomForestClassifier.

    Attributes
    ----------
    _feature_names: list[str]
        The names of the features that are used for classification
    _class_names: list[str]
        The names of the classes. It's an ordered list that is matched to
        annotations [1, 2, 3, ...]
    _classifier: sklearn classifier
        The classifier that is used for classification.
    _training_data_perc: float
        The percentage of the data that is used for training. The rest is used
        for testing.
    _index_columns: list[str]
        The columns that are used for indexing the data.
        Hard-coded to roi_id and label
    _input_schema: pandera.SchemaModel
        The schema for the input data. It's used for validation.
    _schema: pandera.SchemaModel
        Use?
    _predict_schema: pandera.SchemaModel
        The schema for the prediction data.
    _data: pd.DataFrame
        The internal data storage of the classifier. Contains both annotations
        as well as feature measurements for all rows (annotated objects)
    """

    def __init__(self, feature_names, class_names, classifier=RandomForestClassifier()):
        self.logger = logging.getLogger("classifier")
        self.logger.setLevel(logging.INFO)
        self._feature_names: list[str] = list(feature_names)
        self._class_names: list[str] = list(class_names)
        self._classifier = classifier
        self._training_data_perc: float = 0.8
        self._index_columns: list[str] = ["roi_id", "label"]
        (
            self._input_schema,
            self._schema,
            self._predict_schema,
        ) = get_input_internal_and_predict_schemas(
            feature_names=feature_names,
            index_columns=self._index_columns,
            class_names=self._class_names,
        )
        # TODO: `self._schema.example(0)` does not return correct datatypes for
        # MultiIndex (issue: https://github.com/unionai-oss/pandera/issues/1049).
        # Can remove `self._schema.validate` call once fixed.
        self._data: pd.DataFrame = self._schema.validate(self._schema.example(0))

    # TODO: Add tests.
    def train(self):
        """
        Train the classifier on the data it already has in self._data.
        """
        self.logger.info("Training classifier...")
        train_data = self._data[self._data.hash < self._training_data_perc]
        test_data = self._data[self._data.hash >= self._training_data_perc]

        # pylint: disable=C0103
        X_train = train_data.drop(["hash", "annotations"], axis=1)
        # pylint: disable=C0103
        X_test = test_data.drop(["hash", "annotations"], axis=1)

        y_train = train_data["annotations"]
        y_test = test_data["annotations"]

        self._classifier.fit(X_train, y_train)

        f1 = f1_score(y_test, self._classifier.predict(X_test), average="macro")
        self.logger.info(
            f"F1 score on test set: {f1} \n"
            f"Annotations split into {len(X_train)} training and {len(X_test)} "
            "test samples. \n"
            f"Training set contains {self.get_counts_per_class(y_train)}. \n"
            f"Test set contains {self.get_counts_per_class(y_test)}."
        )
        return f1

    def get_counts_per_class(self, y: pd.Series) -> dict[str, int]:
        return {self._class_names[int(k) - 1]: v for k, v in y.value_counts().items()}

    def predict(self, df):
        """
        Run prediction on a single dataframe
        """
        df_valid = self._validate_predict_features(df)
        prediction = self._classifier.predict(df_valid)
        prediction = pd.Series(
            data=prediction, index=df_valid.index, dtype=pd.Int64Dtype()
        ).rename("prediction")
        aligned_prediction = (
            df.reset_index()
            .set_index(self._index_columns)
            .merge(prediction, left_index=True, right_index=True, how="outer")
        )["prediction"]
        return aligned_prediction

    def predict_on_dict(self, dict_of_dfs):
        """
        Loop prediction over a dictionary of dataframes
        """
        # Make a prediction on each of the dataframes provided
        predicted_dicts = {}
        for roi in dict_of_dfs:
            predicted_dicts[roi] = self.predict(dict_of_dfs[roi])
        return predicted_dicts

    def add_features(self, df_raw: pd.DataFrame):
        """
        Add features to the classifier and validate them.
        """
        df_valid = self._validate_input_features(df_raw.reset_index())
        # Select index of annotations to be removed
        index = self._data.index.difference(df_valid.index)
        merged_data = pd.concat([self._data.loc[index], df_valid]).sort_index()

        index_delete = df_valid[df_valid.annotations == -1].index
        self._data = merged_data.drop(index=index_delete)
        return self

    def _validate_predict_features(self, df: pd.DataFrame) -> pd.Series:
        """
        Validate the features that are received for prediction using
        self._predict_schema.
        """
        df_no_nans = df.dropna(subset=self._feature_names)
        if len(df) != len(df_no_nans):
            self.logger.info(
                f"Could not do predictions for {len(df)-len(df_no_nans)}/{len(df)} "
                "objects because of features that contained `NA`s."
            )

        df_valid = self._predict_schema.validate(df_no_nans).set_index(
            self._index_columns
        )
        return df_valid

    def _validate_input_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the newly received training data using self._input_schema.
        """
        # Drop rows that don't have annotations
        df_annotated = df.dropna(subset="annotations")

        # Drop rows that have features with `NA`s, notify the user.
        df_no_nans = df_annotated.dropna(subset=self._feature_names)
        if len(df_no_nans) != len(df_annotated):
            self.logger.info(
                f"Dropped {len(df_annotated)-len(df_no_nans)}/{len(df_annotated)} "
                "objects because of features that contained `NA`s."
            )

        # Validat the dataframe according to the schemas
        df_valid_input = self._input_schema.validate(df_no_nans)
        df_valid_internal = df_valid_input.assign(
            hash=get_normalized_hash_column(df_valid_input, self._index_columns)
        ).set_index(self._index_columns)
        return df_valid_internal

    def add_dict_of_features(self, dict_of_features):
        """
        Add features for each roi

        Parameters
        ----------
        dict_of_features : dict
            Dictionary with roi as key and dataframe with feature measurements
            and annotations as value
        """
        for roi in dict_of_features:
            if "roi_id" not in dict_of_features[roi]:
                dict_of_features[roi]["roi_id"] = roi
            df = dict_of_features[roi]
            self.logger.info(f"Adding features for {roi=}...")
            self.add_features(df)

    def get_class_names(self):
        return self._class_names

    def get_feature_names(self):
        return self._feature_names

    def save(self, output_path):
        self.logger.info(f"Saving classifier at {output_path}...")
        with open(output_path, "wb") as f:
            f.write(pickle.dumps(self))

    def __repr__(self):
        return f"{self.__class__.__name__}\n{repr(self._data)}"


def join_index_columns(df: pd.DataFrame, index_columns: Sequence[str]) -> pd.Series:
    orig_index = df.index
    return pd.Series(
        (
            df.reset_index()
            .loc[:, list(index_columns)]
            .apply(lambda x: "_".join(map(str, x)), axis=1)
        ).values,
        index=orig_index,
    ).rename("object_id")


def hash_single_object_id(object_id: str) -> float:
    max_value = 2**32
    return xxhash.xxh32(object_id).intdigest() / max_value


def get_normalized_hash_column(
    df: pd.DataFrame, index_columns: Sequence[str] = ("roi_id", "label")
) -> pd.Series:
    return join_index_columns(df, index_columns=index_columns).apply(
        hash_single_object_id
    )


def get_random_object_id(n_chars=10):
    return "".join(random.choice(string.ascii_lowercase) for i in range(n_chars))


# TODO: This is cursed, annotations is supposed to be a nullable pandas int type but
# those don't seem to be recognized by pa.Check(ignore_na=True) when doing the checks :(
def get_input_internal_and_predict_schemas(
    feature_names: Sequence[str],
    class_names: Sequence[str],
    index_columns: Sequence[str] = ("roi_id", "label"),
) -> tuple[pa.DataFrameSchema, pa.DataFrameSchema, pa.DataFrameSchema]:
    assert len(index_columns) == 2
    input_schema = pa.DataFrameSchema(
        columns={
            index_columns[0]: pa.Column(pa.String, coerce=True),
            index_columns[1]: pa.Column(pa.UInt64, coerce=True),
            "annotations": pa.Column(
                # pa.Int64,
                pd.Int64Dtype(),
                coerce=True,
                checks=[
                    pa.Check.between(-1, len(class_names)),
                    pa.Check.not_equal_to(0),
                ],
                nullable=True,
            ),
            **{
                feature_name: pa.Column(pa.Float32, coerce=True)
                for feature_name in feature_names
            },
        },
        strict="filter",
        unique=list(index_columns),
    )

    internal_schema = input_schema.set_index(list(index_columns)).add_columns(
        {"hash": pa.Column(pa.Float32, coerce=True, checks=pa.Check.between(0.0, 1.0))}
    )

    predict_schema = input_schema.remove_columns(["annotations"])
    return input_schema, internal_schema, predict_schema
