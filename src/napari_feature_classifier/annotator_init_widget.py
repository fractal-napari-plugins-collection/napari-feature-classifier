import math
import time
import warnings
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, cast

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
    create_widget,
)
from napari.utils.notifications import show_info

from annotator_widget import LabelAnnotator, get_class_selection


def main():
    import imageio

    lbls = imageio.v2.imread("sample_data/test_labels.tif")
    viewer = napari.Viewer()
    viewer.add_labels(lbls)
    viewer.add_labels(lbls, name="lbls2")
    # widget = viewer.window.add_dock_widget(
    #     InitializeLabelAnnotator(
    #         viewer,
    #     )
    # )
    viewer.show(block=True)


class InitializeLabelAnnotator(Container):
    MAX_CLASSES: int = 9

    def __init__(self, viewer: napari.viewer.Viewer, default_n_classes=5):
        self.viewer = viewer
        default_line_edits = [
            LineEdit(value=f"Class_{i + 1}", nullable=True)
            for i in range(default_n_classes)
        ]
        empty_line_edits = [
            LineEdit(nullable=True) for i in range(self.MAX_CLASSES - default_n_classes)
        ]

        self._text_edits = tuple([*default_line_edits, *empty_line_edits])

        self._init_button = PushButton(label="Initialize")
        self._init_button.clicked.connect(self.initialize_annotator)
        super().__init__(widgets=[*self._text_edits, self._init_button])

    def initialize_annotator(self):
        class_names = [e.value for e in self._text_edits if e.value != ""]
        self.viewer.window.add_dock_widget(
            LabelAnnotator(self.viewer, get_class_selection(class_names=class_names))
        )
        init_widget = self.viewer.window._dock_widgets[
            "Init Annotator (napari-feature-classifier)"
        ]
        self.viewer.window.remove_dock_widget(init_widget)


if __name__ == "__main__":
    main()
