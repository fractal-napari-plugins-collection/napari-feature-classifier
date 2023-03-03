import math
import warnings
from enum import Enum
from functools import partial
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
    PushButton,
    RadioButtons,
    TextEdit,
    create_widget,
)

from napari_feature_classifier.annotator_init_widget import LabelAnnotatorTextSelector
from napari_feature_classifier.annotator_widget import (
    LabelAnnotator,
    get_class_selection,
)


def main():
    # lbls = imageio.imread(
    #     r"C:\Users\hessm\Documents\Programming\Python\fractal-napari-plugins\napari-feature-classifier\src\napari_feature_classifier\sample_data\test_labels.tif"
    # )
    viewer = napari.Viewer()
    # viewer.add_labels(lbls)
    widget = ClassifierWidget(viewer)
    viewer.window.add_dock_widget(widget)
    viewer.show(block=True)


class ClassifierInitContainer(Container):
    def __init__(self):
        self._name_edit = TextEdit(value="classifier")
        self._feature_combobox = ComboBox(choices=self.get_feature_options())
        self._annotation_name_selector = LabelAnnotatorTextSelector()
        self._initialize_button = PushButton(text="Initialize")
        super().__init__(
            widgets=[
                self._name_edit,
                self._feature_combobox,
                self._annotation_name_selector,
            ]
        )

    def get_feature_options(self):
        return [f"feature_{i}" for i in range(10)]


class ClassifierRunContainer(Container):
    def __init__(self, viewer: napari.viewer.Viewer, class_names: list[str]):
        self._annotator = LabelAnnotator(viewer, get_class_selection(class_names))
        self._run_button = PushButton(text="Run")
        self._save_button = PushButton(text="Save")
        super().__init__(
            widgets=[
                self._annotator,
                self._run_button,
                self._save_button,
            ]
        )


class ClassifierWidget(Container):
    def __init__(self, viewer: napari.viewer.Viewer):
        self._viewer = viewer
        self._init_container = ClassifierInitContainer()
        super().__init__(widgets=[self._init_container])
        self._init_container._initialize_button.clicked.connect(
            self.initialize_run_widget
        )

    def initialize_run_widget(self):
        pass


if __name__ == "__main__":
    main()
