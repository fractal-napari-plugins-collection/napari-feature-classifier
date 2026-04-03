"""Tests for ClassifierRunner and PredictionLayerManager.

These tests require a napari viewer but have no Qt widget dependency —
they do not use qtbot and can run in a headless environment.
"""

import numpy as np
import pytest

from napari_feature_classifier.classifier import Classifier
from napari_feature_classifier.classifier_runner import (
    ClassifierRunner,
    PredictionLayerManager,
)
from napari_feature_classifier.feature_loader_widget import make_features

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["feature_0", "feature_1", "feature_2"]
CLASS_NAMES = ["ClassA", "ClassB"]
N_LABELS = 20
ROI_ID = "site1"


def make_classifier():
    return Classifier(feature_names=FEATURE_NAMES, class_names=CLASS_NAMES)


def make_label_array(n=N_LABELS):
    """Simple 2-D label image with labels 1..n."""
    img = np.zeros((10, 10), dtype=np.int32)
    for i, label in enumerate(range(1, n + 1)):
        img[i // 10, i % 10] = label
    return img


def make_features_df(roi_id=ROI_ID, n=N_LABELS, with_annotations=False):
    df = make_features(labels=list(range(1, n + 1)), roi_id=roi_id, n_features=3)
    if with_annotations:
        df["annotations"] = [1, 2] * (n // 2)
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def viewer(make_napari_viewer):
    return make_napari_viewer()


@pytest.fixture()
def viewer_with_label_layer(viewer):
    """Viewer with one label layer that has features including roi_id."""
    labels = viewer.add_labels(make_label_array(), name="Labels")
    labels.features = make_features_df()
    return viewer, labels


@pytest.fixture()
def runner(viewer_with_label_layer):
    viewer, _ = viewer_with_label_layer
    return ClassifierRunner(viewer, make_classifier())


@pytest.fixture()
def trained_runner(viewer_with_label_layer):
    """Runner with a trained classifier, ready to predict."""
    viewer, labels = viewer_with_label_layer
    labels.features = make_features_df(with_annotations=True)
    clf = make_classifier()
    runner = ClassifierRunner(viewer, clf)
    runner.add_features_to_classifier()
    clf.train()
    return runner, labels


# ---------------------------------------------------------------------------
# ClassifierRunner: get_relevant_label_layers
# ---------------------------------------------------------------------------


def test_get_relevant_label_layers_returns_layer_with_required_columns(
    viewer_with_label_layer,
):
    viewer, labels = viewer_with_label_layer
    runner = ClassifierRunner(viewer, make_classifier())
    result = runner.get_relevant_label_layers()
    assert labels in result


def test_get_relevant_label_layers_excludes_annotations_layer(viewer):
    labels = viewer.add_labels(make_label_array(), name="Annotations")
    labels.features = make_features_df()
    runner = ClassifierRunner(viewer, make_classifier())
    assert runner.get_relevant_label_layers() == []


def test_get_relevant_label_layers_excludes_predictions_layer(viewer):
    labels = viewer.add_labels(make_label_array(), name="Predictions")
    labels.features = make_features_df()
    runner = ClassifierRunner(viewer, make_classifier())
    assert runner.get_relevant_label_layers() == []


def test_get_relevant_label_layers_requires_roi_id_column(viewer):
    labels = viewer.add_labels(make_label_array(), name="Labels")
    df = make_features_df()
    labels.features = df.drop(columns=["roi_id"])
    runner = ClassifierRunner(viewer, make_classifier())
    assert runner.get_relevant_label_layers() == []


def test_get_relevant_label_layers_requires_label_column(viewer):
    labels = viewer.add_labels(make_label_array(), name="Labels")
    df = make_features_df()
    labels.features = df.rename(columns={"label": "not_label"})
    runner = ClassifierRunner(viewer, make_classifier())
    assert runner.get_relevant_label_layers() == []


# ---------------------------------------------------------------------------
# ClassifierRunner: get_layer_roi_id
# ---------------------------------------------------------------------------


def test_get_layer_roi_id_returns_single_roi_id(viewer_with_label_layer):
    viewer, labels = viewer_with_label_layer
    runner = ClassifierRunner(viewer, make_classifier())
    assert runner.get_layer_roi_id(labels) == ROI_ID


def test_get_layer_roi_id_raises_on_non_unique(viewer):
    labels = viewer.add_labels(make_label_array(), name="Labels")
    df = make_features_df()
    df.loc[df.index[:5], "roi_id"] = "other_site"
    labels.features = df
    runner = ClassifierRunner(viewer, make_classifier())
    with pytest.raises(NotImplementedError):
        runner.get_layer_roi_id(labels)


# ---------------------------------------------------------------------------
# ClassifierRunner: get_relevant_features
# ---------------------------------------------------------------------------


def test_get_relevant_features_returns_correct_columns(viewer_with_label_layer):
    viewer, labels = viewer_with_label_layer
    runner = ClassifierRunner(viewer, make_classifier())
    result = runner.get_relevant_features(labels.features)
    expected_cols = set(FEATURE_NAMES + ["label", "roi_id"])
    assert set(result.columns) == expected_cols


def test_get_relevant_features_filter_annotations_drops_unannotated(viewer):
    labels = viewer.add_labels(make_label_array(), name="Labels")
    df = make_features_df(with_annotations=True)
    # Set some annotations to NaN
    df.loc[df.index[:5], "annotations"] = float("nan")
    labels.features = df
    runner = ClassifierRunner(viewer, make_classifier())
    result = runner.get_relevant_features(labels.features, filter_annotations=True)
    assert result["annotations"].notna().all()
    assert len(result) == N_LABELS - 5


def test_get_relevant_features_set_index(viewer_with_label_layer):
    viewer, labels = viewer_with_label_layer
    runner = ClassifierRunner(viewer, make_classifier())
    result = runner.get_relevant_features(labels.features, set_index=True)
    assert result.index.names == ["roi_id", "label"]


# ---------------------------------------------------------------------------
# ClassifierRunner: add_features_to_classifier
# ---------------------------------------------------------------------------


def test_add_features_to_classifier_only_uses_annotated_layers(viewer):
    labels = viewer.add_labels(make_label_array(), name="Labels")
    labels.features = make_features_df()  # no annotations column
    clf = make_classifier()
    runner = ClassifierRunner(viewer, clf)
    runner.add_features_to_classifier()
    # Classifier internal data should be empty since no layer had annotations
    assert len(clf._data) == 0


def test_add_features_to_classifier_uses_layer_name_when_no_roi_id(viewer):
    labels = viewer.add_labels(make_label_array(), name="my_layer")
    df = make_features_df(with_annotations=True).drop(columns=["roi_id"])
    labels.features = df
    clf = make_classifier()
    runner = ClassifierRunner(viewer, clf)
    runner.add_features_to_classifier()
    assert "my_layer" in clf._data.index.get_level_values("roi_id")


# ---------------------------------------------------------------------------
# ClassifierRunner: make_predictions
# ---------------------------------------------------------------------------


def test_make_predictions_writes_prediction_column(trained_runner):
    runner, labels = trained_runner
    runner.make_predictions()
    assert "prediction" in labels.features.columns


def test_make_predictions_raises_on_duplicate_roi_id(viewer):
    """Two layers with the same roi_id should raise ValueError."""
    for name in ["LayerA", "LayerB"]:
        lyr = viewer.add_labels(make_label_array(), name=name)
        lyr.features = make_features_df(roi_id="same_id")
    clf = make_classifier()
    runner = ClassifierRunner(viewer, clf)
    # Need a trained classifier first; train on LayerA annotations
    viewer.layers["LayerA"].features = make_features_df(
        roi_id="same_id", with_annotations=True
    )
    runner.add_features_to_classifier()
    clf.train()
    # Reset annotations so both layers appear in get_relevant_label_layers
    viewer.layers["LayerA"].features = make_features_df(roi_id="same_id")
    with pytest.raises(ValueError, match="Duplicate roi_id"):
        runner.make_predictions()


# ---------------------------------------------------------------------------
# PredictionLayerManager: init
# ---------------------------------------------------------------------------


def test_prediction_layer_manager_removes_stale_predictions_layer(viewer):
    stale = viewer.add_labels(make_label_array(), name="Predictions")
    assert stale in list(viewer.layers)
    PredictionLayerManager(viewer)
    assert "Predictions" not in [layer.name for layer in viewer.layers]


# ---------------------------------------------------------------------------
# PredictionLayerManager: setup
# ---------------------------------------------------------------------------


def test_setup_creates_predictions_layer(viewer_with_label_layer):
    viewer, labels = viewer_with_label_layer
    mgr = PredictionLayerManager(viewer)
    mgr.setup(labels)
    assert "Predictions" in [layer.name for layer in viewer.layers]


def test_setup_sets_contour(viewer_with_label_layer):
    viewer, labels = viewer_with_label_layer
    mgr = PredictionLayerManager(viewer)
    mgr.setup(labels)
    assert mgr.prediction_layer.contour == 2


def test_setup_syncs_geometry_to_label_layer(viewer):
    labels_a = viewer.add_labels(
        np.zeros((10, 10), dtype=np.int32), name="A", scale=[2.0, 2.0]
    )
    labels_a.features = make_features_df()
    labels_b = viewer.add_labels(
        np.ones((8, 8), dtype=np.int32), name="B", scale=[3.0, 3.0]
    )
    labels_b.features = make_features_df()
    mgr = PredictionLayerManager(viewer)
    mgr.setup(labels_a)
    assert list(mgr.prediction_layer.scale) == [2.0, 2.0]
    mgr.setup(labels_b)
    assert list(mgr.prediction_layer.scale) == [3.0, 3.0]


# ---------------------------------------------------------------------------
# PredictionLayerManager: sync
# ---------------------------------------------------------------------------


def test_sync_does_not_create_layer(viewer_with_label_layer):
    viewer, labels = viewer_with_label_layer
    mgr = PredictionLayerManager(viewer)
    # Call sync without calling setup first
    mgr.sync(labels)
    assert "Predictions" not in [layer.name for layer in viewer.layers]


def test_sync_updates_geometry(viewer_with_label_layer):
    viewer, labels = viewer_with_label_layer
    labels_b = viewer.add_labels(
        np.zeros((6, 6), dtype=np.int32), name="B", scale=[4.0, 4.0]
    )
    labels_b.features = make_features_df()
    mgr = PredictionLayerManager(viewer)
    mgr.setup(labels)
    mgr.sync(labels_b)
    assert list(mgr.prediction_layer.scale) == [4.0, 4.0]
