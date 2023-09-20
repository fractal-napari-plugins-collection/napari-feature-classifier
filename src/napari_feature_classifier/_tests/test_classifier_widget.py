""" Tests for classifier widget initialization"""
import numpy as np
import pandas as pd
import pytest
import os

import imageio
from pathlib import Path
from napari_feature_classifier.feature_loader_widget import make_features
from napari_feature_classifier.classifier_widget import (
    ClassifierWidget,
)
from napari_feature_classifier.classifier_widget import (
    LoadClassifierContainer,
)

lbl_img_np = imageio.v2.imread(
    Path("src/napari_feature_classifier/sample_data/test_labels.tif")
)
clf_path = Path(
    "src/napari_feature_classifier/sample_data/" "test_labels_classifier.clf"
)


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
    assert classifier_widget._init_container._last_selected_label_layer is None


features = make_features(np.unique(lbl_img_np)[1:], roi_id="ROI1", n_features=6)
features_no_roi_id = features.drop(columns=["roi_id"])


# make_napari_viewer is a pytest fixture that returns a napari viewer object
@pytest.mark.parametrize("features", [features, features_no_roi_id])
def test_running_classification_through_widget(features, make_napari_viewer):
    """
    Tests if the main widget launches
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)
    label_layer.features = features

    # Start init widget
    classifier_widget = ClassifierWidget(viewer)

    # Select relevant features
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
    label_layer.features.loc[3, "annotations"] = 3.0

    # Run the classifier
    classifier_widget._run_container.run()

    # Assert something that the layer is visible, predictions exist and are not NaN
    assert classifier_widget._run_container._prediction_layer.visible
    assert "prediction" in label_layer.features.columns
    assert pd.notna(label_layer.features["prediction"]).all().all()

    # Check that the classifier file was saved
    assert Path("lbl_img_np_classifier.clf").exists()

    # Delete the classifier file (cleanup to avoid overwriting confirmation)
    os.remove("lbl_img_np_classifier.clf")


# Test classifier results export
def test_prediction_export(make_napari_viewer, capsys):
    """
    Tests if the main widget launches
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)
    label_layer.features = features

    # Start init widget
    classifier_widget = ClassifierWidget(viewer)

    # Select relevant features
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
    label_layer.features.loc[3, "annotations"] = 3.0

    # Run the classifier
    classifier_widget._run_container.run()

    # Test result export
    classifier_widget._run_container.export_results()
    df = pd.read_csv(classifier_widget._run_container._export_destination.value)
    assert df.shape == (16, 5)
    assert df["prediction"].isna().sum() == 0

    # Delete the classifier file (cleanup to avoid overwriting confirmation)
    os.remove("lbl_img_np_classifier.clf")
    os.remove("lbl_img_np_predictions.csv")

    message = "INFO: Annotations were saved at lbl_img_np_predictions.csv"
    assert message in capsys.readouterr().out


# TODO: Add a test to check the overwrite confirmations working correctly
# For classifier files, for annotations and for exported predictions


features = make_features(np.unique(lbl_img_np)[1:], roi_id="ROI1", n_features=6)


# make_napari_viewer is a pytest fixture that returns a napari viewer object
def test_classifier_fails_running_without_annotation(make_napari_viewer, capsys):
    """
    Tests if the main widget launches
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)
    label_layer.features = features

    # Start init widget
    classifier_widget = ClassifierWidget(viewer)

    # Select relevant features
    classifier_widget._init_container._feature_combobox.value = [
        "feature_1",
        "feature_2",
        "feature_3",
    ]

    classifier_widget.initialize_run_widget()

    # Run the classifier
    classifier_widget._run_container.run()

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
    classifier_widget._init_container._feature_combobox.value = [
        "feature_1",
        "feature_2",
        "feature_3",
    ]

    classifier_widget.initialize_run_widget()

    assert (
        classifier_widget._run_container._last_selected_label_layer.name
        == "test_labels_2"
    )
    assert (
        str(classifier_widget._run_container._export_destination.value)
        == "test_labels_2_predictions.csv"
    )
    viewer.layers.selection.active = label_layer1
    assert (
        classifier_widget._run_container._last_selected_label_layer.name
        == "test_labels"
    )
    assert (
        classifier_widget._run_container._export_destination.value.name
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
    assert loading_widget._run_container._prediction_layer.visible
    assert "prediction" in label_layer.features.columns


def test_change_of_loader_filter(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    loading_widget = LoadClassifierContainer(viewer)
    assert loading_widget._filter.value == "*.clf"
    assert loading_widget._clf_destination.filter is None
    loading_widget._filter.value = "*.pkl"
    assert loading_widget._clf_destination.filter == "*.pkl"
    assert loading_widget._filter.value == "*.pkl"
