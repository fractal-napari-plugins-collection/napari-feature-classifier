"""Helper script to open napari with 2 test label layers with features"""
from pathlib import Path

import imageio
import napari

import numpy as np
from napari_feature_classifier.feature_loader_widget import (
    make_features,
)


def main():
    """Main function that opens napari with 2 test label layers with features"""
    lbls = imageio.v2.imread(Path("sample_data/test_labels.tif"))
    lbls2 = np.zeros_like(lbls)
    lbls2[:, 3:, 2:] = lbls[:, :-3, :-2]
    lbls2 = lbls2 * 20

    labels = np.unique(lbls)[1:]
    labels_2 = np.unique(lbls2)[1:]

    viewer = napari.Viewer()
    lbls_layer = viewer.add_labels(lbls)
    lbls_layer2 = viewer.add_labels(lbls2)

    lbls_layer.features = make_features(labels, roi_id="ROI1", n_features=6)
    lbls_layer2.features = make_features(labels_2, roi_id="ROI2", n_features=6)
    # classifier_widget = ClassifierWidget(viewer)

    # viewer.window.add_dock_widget(classifier_widget)
    viewer.show(block=True)


if __name__ == "__main__":
    main()
