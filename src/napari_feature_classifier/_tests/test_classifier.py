# pylint: disable=C0103
"""Tests for core classifier class"""

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
    counts = c.get_counts_per_class(c._data["annotations"])
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
