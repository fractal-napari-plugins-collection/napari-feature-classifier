""" Tests for classifier widget initialization"""
import numpy as np
import pandas as pd

import imageio
from pathlib import Path
from napari_feature_classifier.feature_loader_widget import make_features
from napari_feature_classifier.classifier_widget import (
    ClassifierWidget,
)

lbl_img_np = imageio.v2.imread(Path("src/napari_feature_classifier/sample_data/test_labels.tif"))

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


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# TODO: Verify the actual results of the classification
def test_running_classification_through_widget(make_napari_viewer):
    """
    Tests if the main widget launches
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)
    labels = np.unique(lbl_img_np)[1:]
    label_layer.features = make_features(labels, roi_id="ROI1", n_features=6)

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

    # TODO: Assert something
    assert classifier_widget._run_container._prediction_layer.visible
    assert "prediction" in label_layer.features.columns
    assert pd.notna(label_layer.features["prediction"]).all().all()
