"""Core classifier class and helper functions."""

import logging
import pickle
import random
import string
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import pandera.pandas as pa
import xxhash
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Seed for the default RandomForestClassifier. Without a fixed seed, every
# retrain on identical annotations gives a different forest and borderline
# objects change class between runs.
DEFAULT_RANDOM_STATE = 0

# Tolerance for deciding whether two feature vectors describe the same object.
# Features are stored as float32, so values are compared after a float32 cast;
# the tolerance only absorbs float noise, any real change of the object fails.
FEATURE_MATCH_RTOL = 1e-5
FEATURE_MATCH_ATOL = 1e-6


def features_match(a, b) -> np.ndarray:
    """
    Row-wise check whether two feature arrays (n_objects x n_features) describe
    the same objects: True where all features are equal after a float32 cast
    (within FEATURE_MATCH_RTOL/ATOL). NaN never matches.
    """
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    close = np.isclose(
        a32, b32, rtol=FEATURE_MATCH_RTOL, atol=FEATURE_MATCH_ATOL, equal_nan=False
    )
    return np.asarray(close.all(axis=1), dtype=bool)


def format_object_ids(index: Iterable, max_shown: int = 20) -> str:
    """Short, human-readable list of (roi_id, label) entries for messages."""
    items = [":".join(map(str, x)) if isinstance(x, tuple) else str(x) for x in index]
    shown = ", ".join(items[:max_shown])
    if len(items) > max_shown:
        shown += f", ... (+{len(items) - max_shown})"
    return shown


class Classifier:
    """Classifier class for napari-feature-classifier.

    Paramters
    ---------
    feature_names: Sequence[str]
        The names of the features that are used for classification
    class_names: Sequence[str]
        The names of the classes. It's an ordered list that is matched to
        annotations [1, 2, 3, ...]
    classifier: sklearn classifier, optional
        The classifier that is used for classification. Default (None) creates
        a new RandomForestClassifier with a fixed random_state for this
        instance, so retraining on the same annotations is reproducible.
        A classifier passed in is used as is.

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

    def __init__(self, feature_names, class_names, classifier=None):
        self.logger = logging.getLogger("classifier")
        self.logger.setLevel(logging.INFO)
        self._feature_names: list[str] = list(feature_names)
        self._class_names: list[str] = list(class_names)
        # Create the default estimator per instance: a default argument would be
        # shared by every Classifier in the session (training one refits all).
        if classifier is None:
            classifier = RandomForestClassifier(random_state=DEFAULT_RANDOM_STATE)
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
        # Per-class colors: keys are 1-based class indices, values are RGBA float tuples.
        # Empty dict means "use Set1 fallback". Persisted via pickle.
        self._class_colors: dict[int, tuple[float, float, float, float]] = {}

    def __setstate__(self, state: dict) -> None:
        """Restore state from pickle, adding defaults for attributes added in later versions."""
        self.__dict__.update(state)
        if "_class_colors" not in self.__dict__:
            self._class_colors = {}
        # Classifiers saved before seeding was introduced have random_state=None,
        # so every retrain produced a different model. Seed them on load. This
        # does not change the already fitted model or its predictions, only
        # makes the next retrain reproducible.
        # Best effort: never fail loading because of this (e.g. non-sklearn
        # estimators without get_params/set_params).
        estimator = self.__dict__.get("_classifier")
        try:
            params = estimator.get_params()  # type: ignore[union-attr]
            if "random_state" in params and params["random_state"] is None:
                estimator.set_params(random_state=DEFAULT_RANDOM_STATE)  # type: ignore[union-attr]
                logging.getLogger("classifier").info(
                    "Loaded classifier had no random_state; set it to "
                    f"{DEFAULT_RANDOM_STATE} so retraining is reproducible."
                )
        except Exception:  # noqa: BLE001
            pass

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
            f"Training set contains {self.get_counts_per_class(y_train)}. \n"  # type: ignore[arg-type]
            f"Test set contains {self.get_counts_per_class(y_test)}."  # type: ignore[arg-type]
        )
        return f1

    def get_counts_per_class(self, y: pd.Series) -> dict[str, int]:
        return {self._class_names[int(k) - 1]: v for k, v in y.value_counts().items()}  # type: ignore[arg-type]

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
        self._warn_if_objects_changed(df_valid)
        # Select index of annotations to be removed
        index = self._data.index.difference(df_valid.index)
        merged_data = pd.concat([self._data.loc[index], df_valid]).sort_index()

        index_delete = df_valid[df_valid.annotations == -1].index
        self._data = merged_data.drop(index=index_delete.tolist())
        return self

    def _warn_if_objects_changed(self, df_new: pd.DataFrame) -> None:
        """
        Warn when incoming rows replace or delete stored rows (same roi_id and
        label) whose features differ, i.e. the label id now refers to a
        different object than the one originally annotated (labels or feature
        table regenerated). Changing only the annotation of an unchanged
        object is a normal re-annotation and does not warn.
        """
        common = df_new.index.intersection(self._data.index)
        if len(common) == 0:
            return
        old = self._data.loc[common, self._feature_names].to_numpy(
            dtype=np.float32, na_value=np.nan
        )
        new = df_new.loc[common, self._feature_names].to_numpy(
            dtype=np.float32, na_value=np.nan
        )
        changed = common[~features_match(old, new)]
        if len(changed) == 0:
            return
        is_delete = (df_new.loc[changed, "annotations"] == -1).to_numpy()
        replaced, deleted = changed[~is_delete], changed[is_delete]
        if len(replaced):
            self.logger.warning(
                f"{len(replaced)} stored annotation(s) were replaced by an "
                "annotation of an object with different features under the same "
                "roi_id and label. The label ids probably refer to different "
                "objects than when they were first annotated (labels or feature "
                "table regenerated?). The previous annotations of these objects "
                f"are no longer used for training: {format_object_ids(replaced)}"
            )
        if len(deleted):
            self.logger.warning(
                f"{len(deleted)} stored annotation(s) were deleted (NoClass) for "
                "label ids that now refer to objects with different features "
                "than the ones originally annotated (labels or feature table "
                f"regenerated?): {format_object_ids(deleted)}"
            )

    def _validate_predict_features(self, df: pd.DataFrame) -> pd.Series:
        """
        Validate the features that are received for prediction using
        self._predict_schema.
        """
        df_no_nans = df.dropna(subset=self._feature_names)
        if len(df) != len(df_no_nans):
            self.logger.info(
                f"Could not do predictions for {len(df) - len(df_no_nans)}/{len(df)} "
                "objects because of features that contained `NA`s."
            )

        df_valid = self._predict_schema.validate(df_no_nans).set_index(
            self._index_columns
        )
        return df_valid  # type: ignore[return-value]

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
                f"Dropped {len(df_annotated) - len(df_no_nans)}/{len(df_annotated)} "
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
    return join_index_columns(df, index_columns=index_columns).apply(  # type: ignore[return-value]
        hash_single_object_id
    )


def get_random_object_id(n_chars=10):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(n_chars))


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
