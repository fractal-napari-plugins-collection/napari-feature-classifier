import math
import warnings
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, cast

import imageio
import matplotlib
import napari
import napari.layers
import napari.viewer
import numpy as np
import pandas as pd
from magicgui.widgets import (
    ComboBox,
    Container,
    FileEdit,
    LineEdit,
    PushButton,
    RadioButtons,
    Select,
    TextEdit,
    create_widget,
)
from napari.utils.notifications import show_info

from napari_feature_classifier.annotator_init_widget import LabelAnnotatorTextSelector
from napari_feature_classifier.annotator_widget import (
    LabelAnnotator,
    get_class_selection,
)


def main():
    lbls = imageio.imread(Path("sample_data/test_labels.tif"))
    lbls2 = np.zeros_like(lbls)
    lbls2[:, 3:, 2:] = lbls[:, :-3, :-2]
    lbls2 = lbls2 * 20

    labels = np.unique(lbls)
    labels2 = np.unique(lbls2)

    viewer = napari.Viewer()
    lbls_layer = viewer.add_labels(lbls)
    lbls_layer2 = viewer.add_labels(lbls2)

    lbls_layer.features = get_features(labels, n_features=6)
    widget = ClassifierWidget(viewer)
    print(widget)
    viewer.window.add_dock_widget(widget)
    viewer.show(block=True)
    dir(lbls_layer.features)


def get_features(labels: Sequence[int], n_features: int = 10, seed: int = 42):
    columns = [f"feature_{i}" for i in range(n_features)]
    rng = np.random.default_rng(seed=seed)
    features = rng.random(size=(len(labels), n_features))
    features[0, :] = np.nan
    return pd.DataFrame(index=labels, columns=columns, data=features)


class ClassifierInitContainer(Container):
    def __init__(self, feature_options: list[str]):
        self._name_edit = LineEdit(value="classifier", label="Classifier Name:")
        self._feature_combobox = Select(
            choices=feature_options, allow_multiple=True, label="Feature Selection:"
        )
        self._annotation_name_selector = LabelAnnotatorTextSelector()
        self._initialize_button = PushButton(text="Initialize")
        super().__init__(
            widgets=[
                self._name_edit,
                self._feature_combobox,
                self._annotation_name_selector,
                self._initialize_button,
            ]
        )


class ClassifierRunContainer(Container):
    def __init__(self, viewer: napari.viewer.Viewer, class_names: list[str]):
        # Optionally get a classifier object or initialize one
        self._annotator = LabelAnnotator(
            viewer, get_class_selection(class_names=class_names)
        )
        self._run_button = PushButton(text="Run Classifier")
        self._save_button = PushButton(text="Save Classifier")
        super().__init__(
            widgets=[
                self._annotator,
                self._run_button,
                self._save_button,
            ]
        )
        self._run_button.clicked.connect(self.run)
        self._save_button.clicked.connect(self.save)

    def run(self):
        # TODO: 
        # 1. Scan all open label layers for annotation & features [ignore annotation layer and predict layer]
        # 2. Update classifier internal feature store
        # 3. Train the classifier
        # 4. Update the prediction layer (create if non-existent) [for one label image => which one]

        show_info("running classifier...")

    def save(self):
        show_info("saving classifier...")


class LoadFeaturesContainer(Container):
    pass


class LoadClassifierContainer(Container):
    # TODO: Implement this. Separate container that leads to run
    # Path to the classifier
    pass


class ClassifierWidget(Container):
    def __init__(self, viewer: napari.viewer.Viewer):
        self._viewer = viewer

        self._init_container = None
        self._run_container = None

        super().__init__(widgets=[])

        self.initialize_init_widget()

    def initialize_init_widget(self):
        # Extract features for first label layer
        # TODO: Handle case where there's no layers.features in the first Labels layer.
        label_layer = [
            l for l in self._viewer.layers if isinstance(l, napari.layers.Labels)
        ][0]
        feature_names = list(label_layer.features.columns)
        self._init_container = ClassifierInitContainer(feature_names)
        self.append(self._init_container)
        self._init_container._initialize_button.clicked.connect(
            self.initialize_run_widget
        )

    def initialize_run_widget(self):
        class_names = self._init_container._annotation_name_selector.get_class_names()
        self._run_container = ClassifierRunContainer(self._viewer, class_names)
        self.clear()
        self.append(self._run_container)


if __name__ == "__main__":
    main()
