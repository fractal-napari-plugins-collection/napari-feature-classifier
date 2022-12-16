from typing import Optional
import numpy as np
import pandas as pd
from magicgui import magic_factory, widgets, magicgui
from napari_plugin_engine import napari_hook_implementation
import napari
from pathlib import Path
from matplotlib.colors import ListedColormap
import matplotlib

from typing import Optional, cast
from enum import Enum

import napari
import napari.layers
import napari.viewer

from magicgui.widgets import Container, create_widget, PushButton, ComboBox, RadioButtons

# Annotator Widget:
# Viewer is open with a label layer

class ClassSelection(Enum):
    NoClass = np.NaN
    Class_1 = 1
    Class_2 = 2
    Class_3 = 3

class LabelAnnotator(Container):
    def __init__(self, viewer: napari.viewer.Viewer):
        self._viewer = viewer
        self._lbl_combo = cast(ComboBox, create_widget(annotation=napari.layers.Labels))
        self._run_btn = PushButton(label="Run")
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._lbl_combo.changed.connect(self._on_label_layer_changed)
        self._annotations_layer: Optional[napari.layers.Labels] = None

        # Class selection
        # self._class_selector = cast(RadioButtons, create_widget(value=0))
        self._class_selector = cast(RadioButtons, create_widget(annotation=ClassSelection, widget_type=RadioButtons))
        #classes={'widget_type': 'RadioButtons', 'choices': [0, 1, 2, 3, 4]}

        super().__init__(widgets=[self._lbl_combo, self._class_selector, self._run_btn])

    def _on_label_layer_changed(self, new_value: napari.layers.Labels):
        print("Label layer changed", new_value)
        # set your internal annotation layer here.

    def _on_run_clicked(self):
        print("Run clicked")
        # whatever you wanted to happen on run


def main():
    import imageio

    lbls = imageio.v2.imread('sample_data/test_labels.tif')
    viewer = napari.Viewer()
    viewer.add_labels(lbls)
    viewer.add_labels(lbls, name='labels2')
    viewer.show(block=True)
#     # print(initialize_annotator())
#     viewer = napari.Viewer()
#     # viewer.add_labels(np.zeros((10, 10), dtype=int))
#     # napari_experimental_provide_dock_widget()
#     viewer.add_labels(np.zeros((10, 10), dtype=np.uint16))
#     # viewer.add_label
#     # threshold_widget = threshold()
#     viewer.window.add_dock_widget(initialize_annotator)
#     viewer.show(block=True)
#     # napari_experimental_provide_dock_widget()


# @magicgui(auto_call=True, threshold={'max': 2 ** 16})
# def threshold(
#     data: 'napari.types.ImageData', threshold: int
# ) -> 'napari.types.LabelsData':
#     return (data > threshold).astype(int)
#
# @napari_hook_implementation
# def napari_experimental_provide_dock_widget():
#     return threshold




if __name__ == '__main__':
    main()
