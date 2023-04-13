from pathlib import Path

from typing import Optional, Sequence, cast

import imageio
import napari
import napari.layers
import napari.viewer
import numpy as np
import pandas as pd
from magicgui.widgets import Container, ComboBox, FileEdit, LineEdit, PushButton, Select, create_widget
from napari.utils.notifications import show_info


from feature_loader_widget import LoadFeaturesContainer, make_features
from napari_feature_classifier.annotator_init_widget import LabelAnnotatorTextSelector
from napari_feature_classifier.annotator_widget import (
    LabelAnnotator,
    get_class_selection,
)
from napari_feature_classifier.classifier_new import Classifier
from napari_feature_classifier.utils import get_colormap, reset_display_colormaps
from napari_feature_classifier.label_layer_selector import LabelLayerSelector

def main():
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
    label_selector_widget = LabelSelector(viewer)

    viewer.window.add_dock_widget(label_selector_widget)
    viewer.show(block=True)


class LabelSelector(Container):
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
    ):
        self._viewer = viewer
        self._lbl_combo = LabelLayerSelector(viewer=self._viewer)
        super().__init__(
            widgets=[
                self._lbl_combo
            ]
        )
        print(f'Selected Layer end ClassifierRunContainer: {self._viewer.layers.selection.active}')


if __name__ == "__main__":
    main()