"""Core classifier class and helper functions."""

import copy
import hashlib
import importlib.metadata
import logging
import pickle
import platform
import random
import string
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import pandera.pandas as pa
import xxhash
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score
from sklearn.utils.validation import check_is_fitted

# Seed for the default RandomForestClassifier. Without a fixed seed, every
# retrain on identical annotations gives a different forest and borderline
# objects change class between runs.
DEFAULT_RANDOM_STATE = 0

# Provenance (versions + events) is stored in the .clf as plain Python types
# only (str/int/float/None/list/dict), so older plugin versions can still load
# the file. The event history is capped to keep files small.
PROVENANCE_PACKAGES = (
    "napari-feature-classifier",
    "scikit-learn",
    "numpy",
    "pandas",
    "pandera",
    "napari",
)
PROVENANCE_HISTORY_MAX = 200


@lru_cache(maxsize=1)
def _environment_versions() -> tuple[tuple[str, str | None], ...]:
    versions: list[tuple[str, str | None]] = [("python", platform.python_version())]
    for package in PROVENANCE_PACKAGES:
        if package == "napari-feature-classifier":
            # Version of the code actually imported (package metadata can be
            # stale, e.g. leftover egg-info on sys.path in dev setups).
            from napari_feature_classifier import __version__ as nfc_version

            if nfc_version != "unknown":
                versions.append((package, nfc_version))
                continue
        try:
            versions.append((package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            versions.append((package, None))
    return tuple(versions)


def environment_versions() -> dict[str, str | None]:
    """Versions of the packages that determine how a classifier behaves."""
    return dict(_environment_versions())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        # Default to INFO, but keep a level the user chose (e.g. DEBUG).
        if self.logger.level == logging.NOTSET:
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
        # Fingerprint of the training data the current model was fitted on.
        # None = unknown / not trained -> the next train_if_needed() retrains.
        self._trained_data_fingerprint: str | None = None
        # When/with which versions this classifier was created, trained, saved.
        self._provenance: dict = {
            "created": {"timestamp": _utc_now(), "versions": environment_versions()},
            "history": [],
        }

    def __setstate__(self, state: dict) -> None:
        """Restore state from pickle, adding defaults for attributes added in later versions."""
        self.__dict__.update(state)
        if "_class_colors" not in self.__dict__:
            self._class_colors = {}
        # Classifiers saved before fingerprints were introduced: it can't be
        # proven which data their model was fitted on, so they count as
        # "needs training" (the first Run retrains once).
        if "_trained_data_fingerprint" not in self.__dict__:
            self._trained_data_fingerprint = None
        # Classifiers saved before provenance was recorded: origin unknown.
        if "_provenance" not in self.__dict__:
            self._provenance = {"created": None, "history": []}
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
                logging.getLogger("classifier").debug(
                    "Loaded classifier had no random_state; set it to "
                    f"{DEFAULT_RANDOM_STATE} so retraining is reproducible."
                )
        except Exception:  # noqa: BLE001
            pass

    def training_data_fingerprint(self) -> str:
        """
        Fingerprint of everything that determines the trained model: all
        stored objects (roi_id, label), their annotations and features, the
        feature order and the train/test split fraction. Independent of row
        order and of pandas dtypes, so re-adding identical annotations does not
        change it.
        """
        data = self._data.sort_index()
        h = hashlib.sha256()
        h.update("\x1f".join(self._feature_names).encode())
        h.update(repr(float(self._training_data_perc)).encode())
        roi_ids = data.index.get_level_values(self._index_columns[0])
        labels = data.index.get_level_values(self._index_columns[1])
        h.update("\x1f".join(map(str, roi_ids)).encode())
        h.update(np.asarray(labels, dtype=np.uint64).tobytes())
        annotations = pd.Series(data["annotations"]).fillna(-99)
        h.update(annotations.to_numpy(dtype=np.int64).tobytes())
        h.update(
            data[self._feature_names]
            .to_numpy(dtype=np.float32, na_value=np.nan)
            .tobytes()
        )
        return h.hexdigest()

    def _model_is_fitted(self) -> bool:
        try:
            check_is_fitted(self._classifier)
        except NotFittedError:
            return False
        except TypeError:  # not an sklearn estimator: rely on the fingerprint
            return True
        return True

    def needs_training(self) -> bool:
        """
        True unless it is certain that the current model was fitted on exactly
        the current training data (same objects, annotations and features).
        Classifiers without a fingerprint (never trained, training failed, or
        saved by an older version) always need training.
        """
        if self._trained_data_fingerprint is None or not self._model_is_fitted():
            return True
        return self._trained_data_fingerprint != self.training_data_fingerprint()

    def train_if_needed(self) -> float | None:
        """
        Retrain only if the training data changed since the model was last
        fitted (see needs_training). Returns the F1 score if it retrained,
        None if the existing model was kept.
        """
        if self.needs_training():
            return self.train()
        self.logger.info(
            "No annotation changes since the last training: predicting with the "
            "existing model (not retrained)."
        )
        return None

    def train(self) -> float:
        """
        Train the classifier on the data it already has in self._data.
        Returns the macro F1 score on the held-out annotations.
        """
        self.logger.debug("Training classifier...")
        # Invalidate first: if anything below fails, the next run retrains.
        self._trained_data_fingerprint = None
        fingerprint = self.training_data_fingerprint()
        train_data = self._data[self._data.hash < self._training_data_perc]
        test_data = self._data[self._data.hash >= self._training_data_perc]

        # pylint: disable=C0103
        X_train = train_data.drop(["hash", "annotations"], axis=1)
        # pylint: disable=C0103
        X_test = test_data.drop(["hash", "annotations"], axis=1)

        y_train = train_data["annotations"]
        y_test = test_data["annotations"]

        self._classifier.fit(X_train, y_train)

        f1 = float(f1_score(y_test, self._classifier.predict(X_test), average="macro"))
        self.logger.info(
            f"Trained on {len(X_train)} annotations "
            f"({self._format_counts(y_train)}). F1 score on {len(X_test)} "  # type: ignore[arg-type]
            f"held-out annotations ({self._format_counts(y_test)}): {f1:.3f}"  # type: ignore[arg-type]
        )
        self._trained_data_fingerprint = fingerprint
        self._record_event(
            "trained",
            n_annotations=len(self._data),
            n_train=len(X_train),
            n_test=len(X_test),
            f1=float(f1),
        )
        return f1

    def _record_event(self, event: str, **details) -> None:
        """Append an event (with timestamp and package versions) to the history."""
        entry = {
            "event": event,
            "timestamp": _utc_now(),
            "versions": environment_versions(),
            **details,
        }
        history = self._provenance.setdefault("history", [])
        history.append(entry)
        del history[:-PROVENANCE_HISTORY_MAX]

    def get_provenance(self) -> dict:
        """
        Copy of the provenance record: "created" (timestamp + package
        versions, None for classifiers saved by versions before 0.5) and
        "history" (trained / saved events with timestamp, versions, details).
        """
        return copy.deepcopy(self._provenance)

    def provenance_summary(self) -> str:
        """One-line, human-readable summary of where this classifier comes from."""

        def describe(entry: dict) -> str:
            v = entry.get("versions", {})
            return (
                f"{entry.get('timestamp', '?')} (napari-feature-classifier "
                f"{v.get('napari-feature-classifier', '?')}, scikit-learn "
                f"{v.get('scikit-learn', '?')})"
            )

        created = self._provenance.get("created")
        parts = [
            f"created {describe(created)}"
            if created
            else "created by a version before 0.5 (no version information stored)"
        ]
        trainings = [
            e for e in self._provenance.get("history", []) if e["event"] == "trained"
        ]
        if trainings:
            parts.append(f"last trained {describe(trainings[-1])}")
        return "Classifier " + ", ".join(parts) + "."

    def get_counts_per_class(self, y: pd.Series) -> dict[str, int]:
        """Number of annotations per class name, in class order."""
        counts = sorted(y.value_counts().items(), key=lambda kv: int(kv[0]))  # type: ignore[arg-type]
        return {self._class_names[int(k) - 1]: int(v) for k, v in counts}  # type: ignore[arg-type]

    def _format_counts(self, y: pd.Series) -> str:
        return ", ".join(f"{k}: {v}" for k, v in self.get_counts_per_class(y).items())

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
            self.logger.debug(f"Adding features for {roi=}...")
            self.add_features(df)

    def get_class_names(self):
        return self._class_names

    def get_feature_names(self):
        return self._feature_names

    def save(self, output_path):
        self.logger.debug(f"Saving classifier at {output_path}...")
        self._record_event("saved", path=str(output_path))
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
