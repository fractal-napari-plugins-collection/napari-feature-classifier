"""Tests for classifier widget initialization"""

from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import pytest

from napari_feature_classifier.classifier_widget import (
    ClassifierWidget,
    LoadClassifierContainer,
)
from napari_feature_classifier.feature_loader_widget import make_features


def run_and_wait(container, qtbot, timeout=30_000):
    """Trigger container.run() and block until the background worker finishes.

    Waits for the run button to be re-enabled, which is the last action in
    both _on_run_done and _on_run_error — guaranteeing those callbacks have
    fully completed regardless of which thread they ran in.
    """
    container.run()
    qtbot.waitUntil(lambda: container._run_button.enabled, timeout=timeout)


lbl_img_np = imageio.v2.imread(
    Path("src/napari_feature_classifier/sample_data/test_labels.tif")
)
clf_path = Path("src/napari_feature_classifier/sample_data/test_labels_classifier.clf")


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_classifier_widgets_initialization_no_features_selected(make_napari_viewer):
    """
    Tests if the main widget launches
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    _ = viewer.add_labels(lbl_img_np)

    # Start init widget
    classifier_widget = ClassifierWidget(viewer)

    classifier_widget.initialize_run_widget()

    # TODO: Catch that it doesn't actually initialize, but instead shows a
    # message to the user that now features were selected
    assert classifier_widget._run_container is None


def test_classifier_initializtion_without_label_image(make_napari_viewer):
    viewer = make_napari_viewer()
    classifier_widget = ClassifierWidget(viewer)
    assert classifier_widget._init_container is not None
    assert classifier_widget._init_container._last_selected_label_layer is None


features = make_features(np.unique(lbl_img_np)[1:], roi_id="ROI1", n_features=6)
features_no_roi_id = features.drop(columns=["roi_id"])


# make_napari_viewer is a pytest fixture that returns a napari viewer object
@pytest.mark.parametrize("features", [features, features_no_roi_id])
def test_running_classification_through_widget(
    features, make_napari_viewer, qtbot, tmp_path, monkeypatch
):
    """
    Tests if the main widget launches
    """
    monkeypatch.chdir(tmp_path)
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)
    label_layer.features = features

    # Start init widget
    classifier_widget = ClassifierWidget(viewer)

    # Select relevant features
    assert classifier_widget._init_container is not None
    classifier_widget._init_container._feature_combobox.value = [
        "feature_1",
        "feature_2",
        "feature_3",
    ]

    classifier_widget.initialize_run_widget()

    # Check that the run widget is initialized
    assert classifier_widget._run_container is not None

    # Add some annotations manually
    label_layer.features.loc[0, "annotations"] = 1.0
    label_layer.features.loc[1, "annotations"] = 1.0
    label_layer.features.loc[3, "annotations"] = 2.0

    # Run the classifier and wait for the background worker to finish
    run_and_wait(classifier_widget._run_container, qtbot)

    # Assert something that the layer is visible, predictions exist and are not NaN
    assert classifier_widget._run_container._prediction_manager.prediction_layer.visible
    assert "prediction" in label_layer.features.columns
    assert pd.notna(label_layer.features["prediction"]).all().all()

    # Check that the classifier file was saved
    assert (tmp_path / "lbl_img_np_classifier.clf").exists()


# Test classifier results export
def test_prediction_export(make_napari_viewer, qtbot, capsys, tmp_path, monkeypatch):
    """
    Tests if the main widget launches
    """
    monkeypatch.chdir(tmp_path)
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)
    label_layer.features = features

    # Start init widget
    classifier_widget = ClassifierWidget(viewer)

    # Select relevant features
    assert classifier_widget._init_container is not None
    classifier_widget._init_container._feature_combobox.value = [
        "feature_1",
        "feature_2",
        "feature_3",
    ]

    classifier_widget.initialize_run_widget()

    # Check that the run widget is initialized
    assert classifier_widget._run_container is not None

    # Add some annotations manually
    label_layer.features.loc[0, "annotations"] = 1.0
    label_layer.features.loc[1, "annotations"] = 1.0
    label_layer.features.loc[3, "annotations"] = 2.0

    # Run the classifier and wait for the background worker to finish
    run_and_wait(classifier_widget._run_container, qtbot)

    # Test result export
    classifier_widget._run_container._export_panel.export_results()
    df = pd.read_csv(classifier_widget._run_container._export_panel._export_path)
    assert df.shape == (16, 5)
    assert df["prediction"].isna().sum() == 0

    message = "INFO: Annotations were saved at lbl_img_np_predictions.csv"
    assert message in capsys.readouterr().out


# TODO: Add a test to check the overwrite confirmations working correctly
# For classifier files, for annotations and for exported predictions


features = make_features(np.unique(lbl_img_np)[1:], roi_id="ROI1", n_features=6)


# make_napari_viewer is a pytest fixture that returns a napari viewer object
def test_classifier_fails_running_without_annotation(
    make_napari_viewer, qtbot, capsys, tmp_path, monkeypatch
):
    """
    Tests if the main widget launches
    """
    monkeypatch.chdir(tmp_path)
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)
    label_layer.features = features

    # Start init widget
    classifier_widget = ClassifierWidget(viewer)

    # Select relevant features
    assert classifier_widget._init_container is not None
    classifier_widget._init_container._feature_combobox.value = [
        "feature_1",
        "feature_2",
        "feature_3",
    ]

    classifier_widget.initialize_run_widget()

    # Run the classifier and wait for the background worker to finish
    assert classifier_widget._run_container is not None
    run_and_wait(classifier_widget._run_container, qtbot)

    expected_message = (
        "INFO: Training failed. A typical reason are not "
        "having enough annotations. \nThe error message was: Found array "
        "with 0 sample(s) (shape=(0, 3)) while a minimum of 1 is required "
        "by RandomForestClassifier."
    )
    assert expected_message in capsys.readouterr().out


def test_layer_selection_changes(make_napari_viewer):
    """
    Tests if the main widget launches
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    label_layer1 = viewer.add_labels(lbl_img_np, name="test_labels")
    label_layer2 = viewer.add_labels(lbl_img_np, name="test_labels_2")
    label_layer2.features = features
    label_layer1.features = features

    # Start init widget
    classifier_widget = ClassifierWidget(viewer)

    # Select relevant features
    assert classifier_widget._init_container is not None
    classifier_widget._init_container._feature_combobox.value = [
        "feature_1",
        "feature_2",
        "feature_3",
    ]

    classifier_widget.initialize_run_widget()

    assert classifier_widget._run_container is not None
    assert (
        classifier_widget._run_container._last_selected_label_layer.name
        == "test_labels_2"
    )
    assert (
        str(classifier_widget._run_container._export_panel._export_path)
        == "test_labels_2_predictions.csv"
    )
    viewer.layers.selection.active = label_layer1
    assert (
        classifier_widget._run_container._last_selected_label_layer.name
        == "test_labels"
    )
    assert (
        classifier_widget._run_container._export_panel._export_path.name
        == "test_labels_predictions.csv"
    )


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_load_classifier_widget(make_napari_viewer, capsys):
    """
    Tests if the load classifier widget launches
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)
    label_layer.features = features

    # Start init widget
    loading_widget = LoadClassifierContainer(viewer)

    loading_widget._clf_destination.value = clf_path
    loading_widget.load()

    # Some basic checks that loading looks to have worked
    assert loading_widget._run_container is not None
    assert loading_widget._run_container._prediction_manager.prediction_layer.visible
    assert "prediction" in label_layer.features.columns


def test_change_of_loader_filter(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    loading_widget = LoadClassifierContainer(viewer)
    assert loading_widget._filter.value == "*.clf"
    assert loading_widget._clf_destination.filter is None
    loading_widget._filter.value = "*.pkl"
    assert loading_widget._clf_destination.filter == "*.pkl"
    assert loading_widget._filter.value == "*.pkl"


def test_rerun_without_changes_does_not_retrain_or_resave(
    make_napari_viewer, qtbot, tmp_path, monkeypatch
):
    """Run with unchanged annotations predicts with the existing model; any
    annotation change retrains (and saves) again."""
    monkeypatch.chdir(tmp_path)
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)
    label_layer.features = features
    classifier_widget = ClassifierWidget(viewer)
    assert classifier_widget._init_container is not None
    classifier_widget._init_container._feature_combobox.value = [
        "feature_1",
        "feature_2",
        "feature_3",
    ]
    classifier_widget.initialize_run_widget()
    container = classifier_widget._run_container
    assert container is not None
    for row, cls in [(0, 1.0), (1, 1.0), (3, 2.0), (4, 2.0)]:
        label_layer.features.loc[row, "annotations"] = cls

    clf = container._classifier
    # Count fits via a class-level patch (an instance attribute would make the
    # classifier unpicklable, and the widget saves it after training).
    from sklearn.ensemble import RandomForestClassifier

    fits = {"n": 0}
    original_fit = RandomForestClassifier.fit

    def counting_fit(self, *args, **kwargs):
        fits["n"] += 1
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(RandomForestClassifier, "fit", counting_fit)
    clf_file = tmp_path / "lbl_img_np_classifier.clf"

    run_and_wait(container, qtbot)
    assert fits["n"] == 1
    assert clf_file.exists()
    mtime = clf_file.stat().st_mtime_ns
    first_predictions = label_layer.features["prediction"].copy()

    # Second run, nothing changed: no refit, file untouched, same predictions
    run_and_wait(container, qtbot)
    assert fits["n"] == 1
    assert clf_file.stat().st_mtime_ns == mtime
    pd.testing.assert_series_equal(
        label_layer.features["prediction"], first_predictions
    )

    # Change of mind on one annotation: retrain + save
    label_layer.features.loc[4, "annotations"] = 1.0
    run_and_wait(container, qtbot)
    assert fits["n"] == 2
    assert clf._trained_data_fingerprint is not None
    assert not clf.needs_training()


def test_loaded_legacy_classifier_retrains_once(
    make_napari_viewer, qtbot, tmp_path, monkeypatch
):
    """A classifier file without fingerprint (older version) retrains on the
    first Run, and not on the next one."""
    import pickle

    from napari_feature_classifier.classifier import Classifier

    # Build a classifier file as an older version would have saved it: trained
    # on this layer's features, but without a training-data fingerprint.
    feature_names = ["feature_1", "feature_2", "feature_3"]
    legacy = Classifier(feature_names, ["Class_1", "Class_2"])
    training = features.copy()
    training["annotations"] = [1, 2] * (len(training) // 2) + [1] * (len(training) % 2)
    legacy.add_features(training)
    legacy.train()
    del legacy.__dict__["_trained_data_fingerprint"]
    legacy_copy = tmp_path / "legacy.clf"
    legacy_copy.write_bytes(pickle.dumps(legacy))
    monkeypatch.chdir(tmp_path)
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)
    label_layer.features = features
    loading_widget = LoadClassifierContainer(viewer)
    loading_widget._clf_destination.value = legacy_copy
    loading_widget.load()
    container = loading_widget._run_container
    assert container is not None
    clf = container._classifier
    assert clf._trained_data_fingerprint is None
    assert clf.needs_training()

    run_and_wait(container, qtbot)
    assert not clf.needs_training()
    mtime = legacy_copy.stat().st_mtime_ns

    run_and_wait(container, qtbot)
    assert legacy_copy.stat().st_mtime_ns == mtime


def test_setup_logging_attaches_one_napari_handler(make_napari_viewer):
    """Opening the widget several times must not duplicate notifications."""
    import logging

    from napari_feature_classifier.utils import NapariHandler

    viewer = make_napari_viewer()
    viewer.add_labels(lbl_img_np).features = features
    ClassifierWidget(viewer)
    ClassifierWidget(viewer)
    logger = logging.getLogger("classifier")
    assert sum(isinstance(h, NapariHandler) for h in logger.handlers) == 1
    # napari echoes notifications to the terminal; no second copy via root
    assert logger.propagate is False


def test_napari_handler_shows_warnings_as_warnings(monkeypatch):
    import logging

    from napari_feature_classifier import utils

    shown = []
    monkeypatch.setattr(utils, "show_info", lambda m: shown.append(("info", m)))
    monkeypatch.setattr(utils, "show_warning", lambda m: shown.append(("warning", m)))
    handler = utils.NapariHandler()
    for level, msg in [(logging.INFO, "hello"), (logging.WARNING, "careful")]:
        handler.emit(logging.LogRecord("classifier", level, "", 0, msg, None, None))
    assert shown == [("info", "hello"), ("warning", "careful")]


def test_loaded_classifier_reports_training_and_skips(
    make_napari_viewer, qtbot, tmp_path, monkeypatch
):
    """Classifiers opened via "Load Classifier" (no ClassifierWidget, no
    Classifier.__init__) must still show the training summary and the
    "not retrained" message."""
    import logging
    import pickle

    from napari.utils.notifications import notification_manager

    from napari_feature_classifier.classifier import Classifier

    logger = logging.getLogger("classifier")
    logger.setLevel(logging.NOTSET)  # as in a fresh napari session
    clf = Classifier(["feature_1", "feature_2", "feature_3"], ["Class_1", "Class_2"])
    logger.setLevel(logging.NOTSET)
    path = tmp_path / "saved.clf"
    path.write_bytes(pickle.dumps(clf))
    monkeypatch.chdir(tmp_path)

    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)
    label_layer.features = features
    loading_widget = LoadClassifierContainer(viewer)
    loading_widget._clf_destination.value = path
    loading_widget.load()
    container = loading_widget._run_container
    assert container is not None
    for row, cls in [(0, 1.0), (1, 1.0), (2, 1.0), (3, 2.0), (4, 2.0), (5, 2.0)]:
        label_layer.features.loc[row, "annotations"] = cls

    def messages_during_run():
        n0 = len(notification_manager.records)
        run_and_wait(container, qtbot)
        return [str(n.message) for n in notification_manager.records[n0:]]

    first = messages_during_run()
    assert any(m.startswith("Trained on ") for m in first)
    assert any(m.startswith("Classifier saved at") for m in first)
    second = messages_during_run()
    assert any("not retrained" in m for m in second)
