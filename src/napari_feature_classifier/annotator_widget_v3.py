import numpy as np
import pandas as pd
from magicgui import magic_factory, widgets, magicgui
from napari_plugin_engine import napari_hook_implementation
import napari
from pathlib import Path
from matplotlib.colors import ListedColormap
import matplotlib
import warnings

from typing import Optional, cast, Sequence
from enum import Enum

import napari
import napari.layers
import napari.viewer

from magicgui.widgets import Container, create_widget, PushButton, ComboBox, RadioButtons, FileEdit

# Annotator Widget:
# Viewer is open with a label layer

class ClassSelection(Enum):
    NoClass = np.NaN
    Class_1 = 1
    Class_2 = 2
    Class_3 = 3
    Class_4 = 4
    
def get_class_selection(n_classes: Optional[int] = 4, class_names: Optional[Sequence[str]] = None) -> Enum:
    if n_classes is None and class_names is None:
        raise ValueError("Provide either `n_classes` or a list of `class_names`")
    if class_names is None:
        class_names = [f'Class_{i+1}' for i in range(n_classes)]
    if n_classes is None:
        n_classes = len(class_names)
    if n_classes != len(class_names):
        warnings.warn(f"Value provided for `n_classes` ({n_classes}) does not match the length of `class_names` ({len(class_names)}). Setting n_classes to {len(class_names)}")
        
    DynamicClassSelection = Enum('DynamicClassSelection', {'NoClass': np.nan, **{c: i+1 for i, c in enumerate(class_names)}})
    return DynamicClassSelection

class LabelAnnotator(Container):
    def __init__(self, viewer: napari.viewer.Viewer, class_selection=get_class_selection(class_names=['mito', 's', 'whatever'])):
        self._viewer = viewer
        self._lbl_combo = cast(ComboBox, create_widget(annotation=napari.layers.Labels))
        self._lbl_combo.changed.connect(self._on_label_layer_changed)
        self._annotations_layer = self._viewer.add_labels(
            self._lbl_combo.value.data, 
            scale=self._lbl_combo.value.scale,
            name='Annotations',
            # editable=False,
        )

        # Class selection
        self._class_selector = cast(RadioButtons, create_widget(value = class_selection[list(class_selection.__members__.keys())[1]], annotation=class_selection, widget_type=RadioButtons))
        self._init_annotation(self._lbl_combo.value)
        self._viewer.layers.selection.events.changed.connect(self._active_changed)
        self._save_destination = FileEdit(value='annotation.csv', mode='r')
        self._save_annotation = PushButton(label="Save Annotations")
        self._save_annotation.clicked.connect(self._on_save_clicked)
        super().__init__(widgets=[
            self._lbl_combo, 
            self._class_selector, 
            self._save_destination, 
            self._save_annotation
        ])

    def _init_annotation(self, label_layer: napari.layers.Labels):
        self._select_layer(label_layer)
        
        if 'annotations' in label_layer.features:
            self.reset_annotation_colormaps()
            return
        unique_labels = np.unique(label_layer.data)[1:]
        label_layer.features['annotations'] = pd.Series([np.NaN]*len(unique_labels), index=unique_labels, dtype=int)
        self.reset_annotation_colormaps()
        #self._annotations_layer = self._viewer.add_labels(self._lbl_combo.value.data) #, scale=self._lbl_combo.scale
        
        
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

            # FIXME: Don't reset the whole colormap, just update a single color
            # Waiting for update on the napari 0.4.17 issue that breaks this option
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
        self._annotations_layer.color = colordict
        self._annotations_layer.opacity=1.0
        self._annotations_layer.color_mode = 'direct'
        
    def _select_layer(self, label_layer: napari.layers.Labels):
        print('selecting layer...')
        self._viewer.layers.selection.clear()
        self._viewer.layers.selection.add(label_layer)
        
    def _active_changed(self, event):
        print('selection changed...')
        current_layer = self._viewer.layers.selection._current
        if type(current_layer) is napari.layers.Labels and current_layer.name != 'Annotations':
            self._lbl_combo.value = self._viewer.layers.selection._current
        else:
            return
        self._lbl_combo.value = self._viewer.layers.selection._current

    def _on_save_clicked(self):
        self._lbl_combo.value.features['annotations'].to_csv(self._save_destination.value)


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


if __name__ == '__main__':
    main()
