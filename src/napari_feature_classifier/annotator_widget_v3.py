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
        self._annotations_layer = self._viewer.add_labels(
            self._lbl_combo.value.data, 
            scale=self._lbl_combo.value.scale,
            name='Annotations'
        )

        # Current label layer: self._lbl_combo.value
        # Class selection
        # self._class_selector = cast(RadioButtons, create_widget(value=0))
        self._class_selector = cast(RadioButtons, create_widget(value = ClassSelection.Class_1, annotation=ClassSelection, widget_type=RadioButtons))
        #classes={'widget_type': 'RadioButtons', 'choices': [0, 1, 2, 3, 4]}
        self._init_annotation(self._lbl_combo.value)
        super().__init__(widgets=[self._lbl_combo, self._class_selector])

    def _init_annotation(self, label_layer: napari.layers.Labels):
        unique_labels = np.unique(label_layer.data)[1:]
        if 'annotations' in label_layer.features:
            return
        label_layer.features['annotations'] = pd.Series([np.NaN]*len(unique_labels), index=unique_labels, dtype=int)
        #self._annotations_layer = self._viewer.add_labels(self._lbl_combo.value.data) #, scale=self._lbl_combo.scale
        self.reset_annotation_colormaps()

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

            label_layer.features['annotations'].loc[label] = self._class_selector.value.value
            print(label_layer.features['annotations'])
            # print(label)
            # print(scaled_position)
            # print(self._class_selector.value)
            # TODO: Trigger color update: Send whole label_layer.features['annotations'] 
            # to the annotation layer colormap
            self.reset_annotation_colormaps()


    def _on_label_layer_changed(self, label_layer: napari.layers.Labels):
        print("Label layer changed", label_layer)
        self._init_annotation(label_layer)
        # set your internal annotation layer here.

    def reset_annotation_colormaps(self):
        """
        Reset the colormap based on the annotations in 
        label_layer.features['annotation'] and sends the updated colormap 
        to the annotation label layer
        """
        cmap = ListedColormap(
            [
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0, 1.0),
                (0.0, 0.0, 1.0, 1.0),
                (1.0, 0.0, 1.0, 1.0),
            ]
        )
        colors = cmap(self._lbl_combo.value.features['annotations'] / 5) # self.nb_classes
        colordict = dict(zip(self._lbl_combo.value.features.index, colors))
        self._annotations_layer.color_mode = 'direct'
        self._annotations_layer.colors = colordict
        print(colordict)
        print(self._annotations_layer.colors)


def main():
    import imageio

    lbls = imageio.v2.imread('sample_data/test_labels.tif')
    viewer = napari.Viewer()
    viewer.add_labels(lbls)
    viewer.add_labels(lbls, name='labels2')
    viewer.show(block=True)


if __name__ == '__main__':
    main()
