# pylint: disable=C0103
"""Tests for core classifier class"""

import logging

import numpy as np
import pandas as pd
import pytest

from napari_feature_classifier.classifier import Classifier

# pylint: disable=C0103
CLASSIFIER_FEATURE_NAMES = ["feature1", "feature2", "feature3"]
DROP_COLUMNS = ["feature6", "feature7"]
TABLE_FEATURE_NAMES = CLASSIFIER_FEATURE_NAMES + DROP_COLUMNS
TABLE_FEATURES = {
    feature_name: np.random.randn(5) for feature_name in TABLE_FEATURE_NAMES
}
CLASS_NAMES = ["s", "m"]
INDEX_COLUMNS = ["roi_id", "label"]


def get_classifier():
    return Classifier(feature_names=CLASSIFIER_FEATURE_NAMES, class_names=CLASS_NAMES)


def get_train_predict_dfs():
    df_train = pd.DataFrame(
        {
            "roi_id": ["site1"] * 50,
            "label": range(1, 50 + 1),
            "annotations": [1, 2, 1, 1, 2] * 10,
            **{
                feature_name: np.random.randn(50)
                for feature_name in TABLE_FEATURE_NAMES
            },
        }
    )
    df_predict = pd.DataFrame(
        {
            "roi_id": ["site2"] * 50,
            "label": range(1, 50 + 1),
            # "annotations": [1, 2, 1, 1, 2] * 10,
            **{
                feature_name: np.random.randn(50)
                for feature_name in TABLE_FEATURE_NAMES
            },
        }
    )
    return df_train, df_predict


def test_prediction():
    df_train, df_predict = get_train_predict_dfs()
    df_predict_with_nans = df_predict.copy()
    df_predict_with_nans.loc[[5, 10, 23], "feature1"] = np.nan
    c = get_classifier()
    c.add_features(df_train)
    c.train()
    predictions = c.predict(df_predict_with_nans)
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(df_predict_with_nans)
    assert set(predictions.dropna().unique()).issubset({1, 2})
    assert predictions.isna().sum() == df_predict_with_nans["feature1"].isna().sum()


def get_df():
    n = 5
    return pd.DataFrame(
        {
            "roi_id": ["site1"] * n,
            "label": range(1, n + 1),
            "annotations": [1, 2, 1, 1, 2],
            **TABLE_FEATURES,
        }
    )


def get_df_index_set():
    return get_df().set_index(INDEX_COLUMNS)


def get_df_with_changed_annotations():
    df_with_changed_annotations = get_df()
    df_with_changed_annotations.loc[
        df_with_changed_annotations.index[:3], "annotations"
    ] = [2, 1, 2]
    return df_with_changed_annotations


def get_df_nan_in_annotations():
    df_nan_in_annotations = get_df()
    df_nan_in_annotations.loc[[1, 4], "annotations"] = np.nan
    df_nan_in_annotations.loc[:, "roi_id"] = "site2"
    return df_nan_in_annotations


def get_df_nan_in_features():
    df_nan_in_features = get_df()
    df_nan_in_features.loc[[0, 1, 3], "feature1"] = np.nan
    df_nan_in_features.loc[:, "roi_id"] = "site3"
    return df_nan_in_features


def get_df_with_columns_to_delete():
    df_with_columns_to_delete = get_df()
    df_with_columns_to_delete.loc[
        df_with_columns_to_delete.index[:3], "annotations"
    ] = -1
    return df_with_columns_to_delete


def test_can_be_initialized_and_is_empty_upon_init():
    c = get_classifier()
    assert len(c._data) == 0


@pytest.mark.parametrize("df", [get_df(), get_df_index_set()])
def test_can_add_feature_table(df):
    c = get_classifier()
    df = get_df()
    c.add_features(df)
    assert len(c._data) == len(df)
    assert all(feature in c._data for feature in CLASSIFIER_FEATURE_NAMES)
    assert not any(feature in c._data for feature in DROP_COLUMNS)
    assert "hash" in c._data


def test_adding_the_same_table_multiple_times_has_no_effect():
    c1 = get_classifier()
    c2 = get_classifier()
    df = get_df()
    df_index = get_df_index_set()

    c1.add_features(df)
    c2.add_features(df_index)
    assert np.all(c1._data == c2._data)

    for _ in range(5):
        c2.add_features(df)
    assert np.all(c1._data == c2._data)


def test_rows_with_nan_in_annotation_not_added():
    c = get_classifier()
    df_nan = get_df_nan_in_annotations()

    c.add_features(df_nan)
    assert len(c._data == 3)
    assert np.all(c._data.annotations == c._data.annotations.dropna())


def test_rows_with_nan_in_features_not_added():
    c = get_classifier()
    df_nan = get_df_nan_in_features()

    c.add_features(df_nan)
    assert len(c._data) == 2
    assert np.all(c._data.feature1 == c._data.feature1.dropna())


def test_can_change_annotations():
    c = get_classifier()
    df = get_df()
    df2 = get_df_with_changed_annotations()
    assert np.any(df.annotations != df2.annotations)

    c.add_features(df)
    assert np.all(c._data.annotations.values == df.annotations.values)
    c.add_features(df2)
    assert np.all(c._data.annotations.values == df2.annotations.values)


def test_can_delete_annotations():
    c = get_classifier()
    df = get_df()
    df2 = get_df_with_columns_to_delete()

    c.add_features(df)
    assert len(c._data) == 5
    c.add_features(df2)
    assert len(c._data) == 2


def test_add_multi_site_with_changes_and_deletes():
    c = get_classifier()
    df = get_df()
    df_nan = get_df_nan_in_annotations()
    df_nan_feat = get_df_nan_in_features()
    df_change = get_df_with_changed_annotations()
    df_delete = get_df_with_columns_to_delete()

    c.add_features(df)
    assert len(c._data) == 5
    c.add_features(df_nan)
    assert len(c._data) == 8
    c.add_features(df_nan_feat)
    assert len(c._data) == 10
    c.add_features(df_change)
    assert len(c._data) == 10
    c.add_features(df_delete)
    assert len(c._data) == 7


def test_nans_in_non_classifier_features_have_no_effect():
    c = get_classifier()
    c2 = get_classifier()
    df = get_df()
    df_nans = df.copy()
    df_nans.loc[[1, 3, 4], "feature6"] = np.nan

    c.add_features(df)
    c2.add_features(df_nans)
    assert np.all(c._data == c2._data)


def test_train_returns_f1_score():
    df_train, _ = get_train_predict_dfs()
    c = get_classifier()
    c.add_features(df_train)
    f1 = c.train()
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0


def test_train_produces_predictions():
    """After training, predict() returns valid class labels for all non-NaN rows."""
    df_train, df_predict = get_train_predict_dfs()
    c = get_classifier()
    c.add_features(df_train)
    c.train()
    predictions = c.predict(df_predict)
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(df_predict)
    assert set(predictions.dropna().unique()).issubset({1, 2})
    assert predictions.isna().sum() == 0


def test_train_splits_data_correctly():
    """Training uses 80% of data; test split is the remaining 20%."""
    df_train, _ = get_train_predict_dfs()
    c = get_classifier()
    c.add_features(df_train)
    # Verify data is split by hash threshold
    train_data = c._data[c._data.hash < c._training_data_perc]
    test_data = c._data[c._data.hash >= c._training_data_perc]
    assert len(train_data) + len(test_data) == len(df_train)
    assert len(train_data) > 0
    assert len(test_data) > 0


def test_get_counts_per_class():
    df_train, _ = get_train_predict_dfs()
    c = get_classifier()
    c.add_features(df_train)
    counts = c.get_counts_per_class(pd.Series(c._data["annotations"]))
    assert set(counts.keys()) == set(CLASS_NAMES)
    assert counts["s"] == 30  # annotations=[1,2,1,1,2]*10 → class 1 ("s") = 30
    assert counts["m"] == 20


def test_save_and_load_roundtrip(tmp_path):
    """save() pickles correctly; loading restores state including _class_colors."""
    import pickle

    df_train, df_predict = get_train_predict_dfs()
    c = get_classifier()
    c.add_features(df_train)
    c.train()
    c._class_colors[1] = (1.0, 0.0, 0.0, 1.0)

    path = tmp_path / "test.clf"
    c.save(path)

    with open(path, "rb") as f:
        c2 = pickle.loads(f.read())  # noqa: S301

    assert c2._feature_names == c._feature_names
    assert c2._class_names == c._class_names
    assert c2._class_colors == {1: (1.0, 0.0, 0.0, 1.0)}
    # Loaded classifier should predict the same values
    pred_orig = c.predict(df_predict)
    pred_loaded = c2.predict(df_predict)
    assert (pred_orig == pred_loaded).all()


def test_setstate_adds_class_colors_default():
    """Old pickled classifiers without _class_colors get an empty dict on load."""
    import pickle

    df_train, _ = get_train_predict_dfs()
    c = get_classifier()
    c.add_features(df_train)
    c.train()

    # Simulate an old pickle by removing _class_colors before pickling
    state = c.__dict__.copy()
    del state["_class_colors"]
    c.__dict__ = state

    data = pickle.dumps(c)  # noqa: S301
    c2 = pickle.loads(data)  # noqa: S301
    assert c2._class_colors == {}


# ---------------------------------------------------------------------------
# Seeding and per-instance default estimator
# ---------------------------------------------------------------------------


def test_default_estimator_is_not_shared_between_instances():
    """Each Classifier gets its own estimator (no shared mutable default)."""
    a, b = get_classifier(), get_classifier()
    assert a._classifier is not b._classifier


def test_training_one_classifier_does_not_affect_another():
    df_train, df_predict = get_train_predict_dfs()
    a = get_classifier()
    a.add_features(df_train)
    a.train()
    before = a.predict(df_predict)

    other_features = ["feature6", "feature7"]
    b = Classifier(feature_names=other_features, class_names=CLASS_NAMES)
    b.add_features(df_train)
    b.train()

    pd.testing.assert_series_equal(a.predict(df_predict), before)


def test_default_estimator_is_seeded():
    from napari_feature_classifier.classifier import DEFAULT_RANDOM_STATE

    c = get_classifier()
    assert c._classifier.get_params()["random_state"] == DEFAULT_RANDOM_STATE


def test_retraining_on_same_annotations_is_reproducible():
    df_train, df_predict = get_train_predict_dfs()
    preds = []
    classifiers = [get_classifier() for _ in range(3)]
    for c in classifiers:
        c.add_features(df_train)
        c.train()
        preds.append(c.predict(df_predict))
    # retraining the same instance as well
    classifiers[-1].train()
    preds.append(classifiers[-1].predict(df_predict))
    for p in preds[1:]:
        pd.testing.assert_series_equal(p, preds[0])


def test_custom_estimator_is_used_as_is():
    from sklearn.ensemble import RandomForestClassifier

    est = RandomForestClassifier(n_estimators=7, random_state=123)
    c = Classifier(CLASSIFIER_FEATURE_NAMES, CLASS_NAMES, classifier=est)
    assert c._classifier is est
    assert c._classifier.get_params()["random_state"] == 123


def test_setstate_seeds_old_unseeded_classifier_without_changing_predictions():
    """Pickles from before seeding (random_state=None) get seeded on load; the
    fitted model and its predictions are unchanged."""
    import pickle

    from sklearn.ensemble import RandomForestClassifier

    from napari_feature_classifier.classifier import DEFAULT_RANDOM_STATE

    df_train, df_predict = get_train_predict_dfs()
    old = Classifier(
        CLASSIFIER_FEATURE_NAMES, CLASS_NAMES, classifier=RandomForestClassifier()
    )
    old.add_features(df_train)
    old.train()
    before = old.predict(df_predict)

    loaded = pickle.loads(pickle.dumps(old))  # noqa: S301
    assert loaded._classifier.get_params()["random_state"] == DEFAULT_RANDOM_STATE
    pd.testing.assert_series_equal(loaded.predict(df_predict), before)


def test_setstate_keeps_existing_seed():
    import pickle

    from sklearn.ensemble import RandomForestClassifier

    c = Classifier(
        CLASSIFIER_FEATURE_NAMES,
        CLASS_NAMES,
        classifier=RandomForestClassifier(random_state=42),
    )
    loaded = pickle.loads(pickle.dumps(c))  # noqa: S301
    assert loaded._classifier.get_params()["random_state"] == 42


class _EstimatorWithoutParams:
    """Minimal non-sklearn estimator (no get_params/set_params)."""

    def fit(self, _, y):
        self.value_ = int(pd.Series(y).mode()[0])
        return self

    def predict(self, X):
        return np.full(len(X), getattr(self, "value_", 1))


def test_setstate_loads_classifier_with_non_sklearn_estimator():
    import pickle

    c = Classifier(
        CLASSIFIER_FEATURE_NAMES, CLASS_NAMES, classifier=_EstimatorWithoutParams()
    )
    loaded = pickle.loads(pickle.dumps(c))  # noqa: S301
    assert isinstance(loaded._classifier, _EstimatorWithoutParams)


# ---------------------------------------------------------------------------
# Warnings when a stored (roi_id, label) now refers to a different object
# ---------------------------------------------------------------------------


def _stored_classifier():
    """Classifier with 50 stored annotations (site1, labels 1..50)."""
    df_train, _ = get_train_predict_dfs()
    c = get_classifier()
    c.add_features(df_train)
    return c, df_train


def _changed_object_warnings(caplog):
    return [r for r in caplog.records if r.levelname == "WARNING"]


def test_reannotating_unchanged_object_does_not_warn(caplog):
    """Users may change their mind about an object's class: no warning."""
    c, df = _stored_classifier()
    df2 = df.copy()
    df2["annotations"] = 3 - df2["annotations"]  # swap classes 1 <-> 2
    with caplog.at_level("WARNING", logger="classifier"):
        c.add_features(df2)
    assert _changed_object_warnings(caplog) == []
    assert (c._data["annotations"].to_numpy() == df2["annotations"].to_numpy()).all()


def test_re_adding_identical_float64_table_does_not_warn(caplog):
    """Stored float32 vs incoming float64 of the same objects is not a change."""
    c, df = _stored_classifier()
    with caplog.at_level("WARNING", logger="classifier"):
        c.add_features(df)
    assert _changed_object_warnings(caplog) == []


def test_replacing_row_of_changed_object_warns_and_still_replaces(caplog):
    c, df = _stored_classifier()
    df2 = df.iloc[[3, 4]].copy()
    df2.loc[df2.index[0], "feature1"] += 1.0  # label 4 is now another object
    with caplog.at_level("WARNING", logger="classifier"):
        c.add_features(df2)
    warnings_ = _changed_object_warnings(caplog)
    assert len(warnings_) == 1
    assert "1 stored annotation(s) were replaced" in warnings_[0].getMessage()
    assert "site1:4" in warnings_[0].getMessage()
    assert "site1:5" not in warnings_[0].getMessage()
    assert len(c._data) == 50
    stored_value = float(c._data.loc[("site1", 4), "feature1"])  # type: ignore[arg-type]
    assert np.isclose(stored_value, np.float32(df2.iloc[0]["feature1"]))


def test_noclass_on_changed_object_warns_and_still_deletes(caplog):
    c, df = _stored_classifier()
    df2 = df.iloc[[5]].copy()
    df2["annotations"] = -1
    df2["feature2"] += 1.0
    with caplog.at_level("WARNING", logger="classifier"):
        c.add_features(df2)
    warnings_ = _changed_object_warnings(caplog)
    assert len(warnings_) == 1
    assert "were deleted (NoClass)" in warnings_[0].getMessage()
    assert ("site1", 6) not in c._data.index


def test_noclass_on_unchanged_object_does_not_warn(caplog):
    c, df = _stored_classifier()
    df2 = df.iloc[[5]].copy()
    df2["annotations"] = -1
    with caplog.at_level("WARNING", logger="classifier"):
        c.add_features(df2)
    assert _changed_object_warnings(caplog) == []
    assert ("site1", 6) not in c._data.index


def test_new_objects_do_not_warn(caplog):
    c, df = _stored_classifier()
    df2 = df.copy()
    df2["roi_id"] = "site2"
    with caplog.at_level("WARNING", logger="classifier"):
        c.add_features(df2)
    assert _changed_object_warnings(caplog) == []
    assert len(c._data) == 100


# ---------------------------------------------------------------------------
# Retrain only when the training data changed (fingerprint)
# ---------------------------------------------------------------------------


def _trained_classifier():
    df_train, df_predict = get_train_predict_dfs()
    c = get_classifier()
    c.add_features(df_train)
    c.train()
    return c, df_train, df_predict


def _count_fits(monkeypatch, c):
    calls = {"n": 0}
    original_fit = c._classifier.fit

    def counting_fit(*args, **kwargs):
        calls["n"] += 1
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(c._classifier, "fit", counting_fit)
    return calls


def test_new_classifier_needs_training():
    c = get_classifier()
    df_train, _ = get_train_predict_dfs()
    c.add_features(df_train)
    assert c.needs_training()


def test_trained_classifier_does_not_need_training():
    c, _, _ = _trained_classifier()
    assert not c.needs_training()


def test_train_if_needed_skips_fit_when_nothing_changed(monkeypatch):
    c, df_train, df_predict = _trained_classifier()
    before = c.predict(df_predict)
    calls = _count_fits(monkeypatch, c)
    c.add_features(df_train)  # re-adding identical (float64) annotations
    assert not c.needs_training()
    assert c.train_if_needed() is None
    assert calls["n"] == 0
    pd.testing.assert_series_equal(c.predict(df_predict), before)


@pytest.mark.parametrize("change", ["relabel", "new_object", "delete", "feature"])
def test_any_change_to_training_data_requires_retraining(monkeypatch, change):
    c, df_train, _ = _trained_classifier()
    df2 = df_train.iloc[[0]].copy()
    if change == "relabel":
        df2["annotations"] = 3 - df2["annotations"]
    elif change == "new_object":
        df2["label"] = 999
    elif change == "delete":
        df2["annotations"] = -1
    elif change == "feature":
        df2["feature1"] += 1.0
    c.add_features(df2)
    assert c.needs_training()
    calls = _count_fits(monkeypatch, c)
    assert c.train_if_needed() is not None
    assert calls["n"] == 1
    assert not c.needs_training()


def test_failed_training_forces_retraining_next_time(monkeypatch):
    c, df_train, _ = _trained_classifier()

    def failing_fit(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(c._classifier, "fit", failing_fit)
    with pytest.raises(ValueError):
        c.train()
    assert c._trained_data_fingerprint is None
    assert c.needs_training()


def test_fingerprint_survives_save_and_load():
    import pickle

    c, _, _ = _trained_classifier()
    loaded = pickle.loads(pickle.dumps(c))  # noqa: S301
    assert not loaded.needs_training()


def test_classifier_from_older_version_needs_training_once():
    """Files without a fingerprint (saved before this feature) can't prove
    which data their model was trained on: retrain once, then not again."""
    import pickle

    c, _, _ = _trained_classifier()
    del c.__dict__["_trained_data_fingerprint"]
    legacy = pickle.loads(pickle.dumps(c))  # noqa: S301
    assert legacy._trained_data_fingerprint is None
    assert legacy.needs_training()
    assert legacy.train_if_needed() is not None
    assert not legacy.needs_training()


def test_fingerprint_independent_of_row_order():
    c, df_train, _ = _trained_classifier()
    fp = c.training_data_fingerprint()
    c2 = get_classifier()
    c2.add_features(df_train.iloc[::-1])
    assert c2.training_data_fingerprint() == fp


# ---------------------------------------------------------------------------
# Provenance (versions, timestamps, events) stored in the classifier
# ---------------------------------------------------------------------------


def _only_builtin_types(obj) -> bool:
    """Provenance must only contain plain Python types so that older plugin
    versions (without the code that wrote it) can still unpickle the file."""
    if isinstance(obj, dict):
        return all(
            isinstance(k, str) and _only_builtin_types(v) for k, v in obj.items()
        )
    if isinstance(obj, list):
        return all(_only_builtin_types(v) for v in obj)
    return obj is None or type(obj) in (str, int, float, bool)


def test_new_classifier_records_creation_versions():
    from datetime import datetime

    c = get_classifier()
    created = c.get_provenance()["created"]
    assert datetime.fromisoformat(created["timestamp"]).tzinfo is not None
    versions = created["versions"]
    for package in ["python", "napari-feature-classifier", "scikit-learn", "numpy"]:
        assert versions[package]
    assert c.get_provenance()["history"] == []


def test_training_and_saving_are_recorded(tmp_path):
    import pickle

    c, df_train, _ = _trained_classifier()
    path = tmp_path / "c.clf"
    c.save(path)
    loaded = pickle.loads(path.read_bytes())  # noqa: S301
    history = loaded.get_provenance()["history"]
    assert [e["event"] for e in history] == ["trained", "saved"]
    trained = history[0]
    assert trained["n_annotations"] == len(df_train)
    assert trained["n_train"] + trained["n_test"] == len(df_train)
    assert 0.0 <= trained["f1"] <= 1.0
    assert history[1]["path"] == str(path)
    assert trained["versions"]["scikit-learn"]


def test_skipped_training_is_not_recorded():
    c, df_train, _ = _trained_classifier()
    c.add_features(df_train)
    c.train_if_needed()
    assert [e["event"] for e in c.get_provenance()["history"]] == ["trained"]


def test_provenance_contains_only_builtin_types(tmp_path):
    c, _, _ = _trained_classifier()
    c.save(tmp_path / "c.clf")
    assert _only_builtin_types(c._provenance)


def test_provenance_history_is_capped():
    from napari_feature_classifier.classifier import PROVENANCE_HISTORY_MAX

    c = get_classifier()
    for i in range(PROVENANCE_HISTORY_MAX + 25):
        c._record_event("trained", n_annotations=i)
    history = c.get_provenance()["history"]
    assert len(history) == PROVENANCE_HISTORY_MAX
    assert history[-1]["n_annotations"] == PROVENANCE_HISTORY_MAX + 24


def test_get_provenance_returns_a_copy():
    c = get_classifier()
    c.get_provenance()["history"].append({"event": "fake"})
    assert c.get_provenance()["history"] == []


def test_classifier_from_older_version_has_unknown_provenance():
    import pickle

    c, _, _ = _trained_classifier()
    del c.__dict__["_provenance"]
    legacy = pickle.loads(pickle.dumps(c))  # noqa: S301
    assert legacy.get_provenance() == {"created": None, "history": []}
    assert "before 0.5" in legacy.provenance_summary()
    legacy.train()
    assert [e["event"] for e in legacy.get_provenance()["history"]] == ["trained"]


def test_provenance_summary_mentions_versions():
    c, _, _ = _trained_classifier()
    summary = c.provenance_summary()
    assert "scikit-learn" in summary
    assert "last trained" in summary


# ---------------------------------------------------------------------------
# Logging: one concise message per user-relevant event
# ---------------------------------------------------------------------------


def _info_messages(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "classifier" and r.levelno == logging.INFO
    ]


def test_training_logs_a_single_concise_summary(caplog, tmp_path):
    df_train, _ = get_train_predict_dfs()
    c = get_classifier()
    with caplog.at_level(logging.INFO, logger="classifier"):
        c.add_dict_of_features({"site1": df_train})
        c.train()
        c.save(tmp_path / "c.clf")
    messages = _info_messages(caplog)
    assert len(messages) == 1
    assert messages[0].startswith("Trained on ")
    assert "np.int64" not in messages[0]
    # classes listed in class order
    assert messages[0].index("s: ") < messages[0].index("m: ")


def test_skipped_training_logs_one_message(caplog):
    c, df_train, _ = _trained_classifier()
    c.add_features(df_train)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="classifier"):
        c.train_if_needed()
    messages = _info_messages(caplog)
    assert len(messages) == 1
    assert "not retrained" in messages[0]


def test_get_counts_per_class_returns_plain_ints_in_class_order():
    c, _, _ = _trained_classifier()
    counts = c.get_counts_per_class(pd.Series([2, 1, 2, 2]))
    assert list(counts) == ["s", "m"]
    assert counts == {"s": 1, "m": 3}
    assert all(type(v) is int for v in counts.values())
