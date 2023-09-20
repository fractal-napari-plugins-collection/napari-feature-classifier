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

lbl_img_np = imageio.v2.imread(Path("src/napari_feature_classifier/sample_data/test_labels.tif"))
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

# TODO: Add a test to check the overwrite confirmations working correctly
# For classifier files, for annotations and for exported predictions

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
    assert loading_widget._clf_destination.filter == None
    loading_widget._filter.value = "*.pkl"
    assert loading_widget._clf_destination.filter == "*.pkl"
    assert loading_widget._filter.value == "*.pkl"




