""" Tests feature loading"""
import pandas as pd

import imageio
from pathlib import Path
from napari_feature_classifier.feature_loader_widget import (
    load_features_factory,
)

lbl_img_np = imageio.v2.imread(
    Path("src/napari_feature_classifier/sample_data/test_labels.tif")
)
csv_path = Path("src/napari_feature_classifier/sample_data/test_df.csv")


def test_feature_loading_csv(make_napari_viewer, capsys):
    """
    Tests if the main widget launches
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    labels_layer = viewer.add_labels(lbl_img_np)

    # Start feature loading widget
    loading_widget = load_features_factory()
    loading_widget.layer.value = labels_layer
    loading_widget.path.value = csv_path
    features = pd.read_csv(csv_path)
    loading_widget.__call__()
    assert (labels_layer.features == features).all().all()

    # Assert that this message is logged
    expected = "INFO: Loaded features and attached them "
    expected += 'to "lbl_img_np" layer\n'
    assert expected in capsys.readouterr().out
