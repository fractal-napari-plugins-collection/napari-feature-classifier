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
        self._lbl_combo.changed.connect(self._on_label_layer_changed)
        self._annotations_layer: Optional[napari.layers.Labels] = None

        # Current label layer: self._lbl_combo.value
        # Class selection
        # self._class_selector = cast(RadioButtons, create_widget(value=0))
        self._class_selector = cast(RadioButtons, create_widget(value = ClassSelection.Class_1, annotation=ClassSelection, widget_type=RadioButtons))
        #classes={'widget_type': 'RadioButtons', 'choices': [0, 1, 2, 3, 4]}
        self._init_annotation(self._lbl_combo.value)
        super().__init__(widgets=[self._lbl_combo, self._class_selector])

    def _init_annotation(self, label_layer: napari.layers.Labels):
        self._select_layer(label_layer)
        
        unique_labels = np.unique(label_layer.data)[1:]
        if 'annotations' in label_layer.features:
            return
        label_layer.features['annotations'] = pd.Series([np.NaN]*len(unique_labels), index=unique_labels, dtype=int)
        
        @self._lbl_combo.value.mouse_drag_callbacks.append
        def toggle_label(labels_layer, event):  # pylint: disable-msg=W0613
            # Need to scale position that event.position returns by the
            # label_layer scale.
            # If scale is (1, 1, 1), nothing changes
            # If scale is anything else, this makes the click still match the
            # correct label
            scaled_position = tuple(
                pos / scale for pos, scale in zip(event.position, labels_layer.scale)
            )
            label = labels_layer.get_value(scaled_position)
            if label == 0:
                print('No label clicked.')
                return

            label_layer.features['annotations'] .loc[label] = self._class_selector.value.value
            print(label_layer.features['annotations'])
            # print(label)
            # print(scaled_position)
            # print(self._class_selector.value)

    def _on_label_layer_changed(self, label_layer: napari.layers.Labels):
        print("Label layer changed", label_layer)
        self._init_annotation(label_layer)
        # set your internal annotation layer here.
        
    def _select_layer(self, label_layer: napari.layers.Labels):
        print('selecting layer...')
        self._viewer.layers.selection.clear()
        self._viewer.layers.selection.add(label_layer)

    #     """
    #     Handles user annotations by setting the corresponding classifier
    #     variables and changing the annotation label layer
    #     """
    #     annotation_layer.visible=True
    #     # Need to scale position that event.position returns by the
    #     # label_layer scale.
    #     # If scale is (1, 1, 1), nothing changes
    #     # If scale is anything else, this makes the click still match the
    #     # correct label
    #     scaled_position = tuple(
    #         pos / scale for pos, scale in zip(event.position, label_layer.scale)
    #     )
    #     label = label_layer.get_value(scaled_position)
    #     if classes is None:
    #         print(
    #             "No class is selected. Select a class in the classifier widget."
    #         )
    #         return

    #     # Check if background or foreground was clicked. If background was
    #     # clicked, do nothing (background can't be assigned a class)
    #     if label == 0 or label is None:
    #         print("No label clicked.")
    #         return

    #     # TODO: Handle the "0" case => np.Nan
    #     if classes == 0:
    #         label_layer.features.loc[label, "annotation"] = np.NaN
    #     else:
    #         label_layer.features.loc[label, "annotation"] = int(classes)        

    #     # TODO: Need to have colormaps initialized before using them here
    #     print(classes)
    #     print(cmap(int(classes)))
    #     # Problem: Figure out how colors are direct assigned in napari v0.4.17
    #     annotation_layer.color[label] = cmap(int(classes))
    #     annotation_layer.color_mode = 'direct'


def main():
    import imageio

    lbls = imageio.v2.imread('sample_data/test_labels.tif')
    viewer = napari.Viewer()
    viewer.add_labels(lbls)
    viewer.add_labels(lbls, name='lbls2')
    widget = viewer.window.add_dock_widget(LabelAnnotator(viewer))
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
