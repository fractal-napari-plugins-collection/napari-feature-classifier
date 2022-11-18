"""Annotator widget code."""
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import h5py
import matplotlib
import napari
import numpy as np
import pandas as pd
from magicgui import widgets
from matplotlib.colors import ListedColormap
from napari.layers.labels import Labels
from qtpy.QtWidgets import QMessageBox  # pylint: disable-msg=E0611

from src.napari_feature_classifier.utils import napari_info


def main():
    img = np.random.randint(255, size=(100, 100))
    lbl = np.zeros_like(img)
    lbl[10:20, 10:20] = 1
    lbl[40:50, 40:50] = 2
    lbl[30:40, 80:90] = 4
    # fn = Path(r"Z:\hmax\Zebrafish\20211119_cyclerDATA_compressed\20211119_ABpanelTestset_3pair_3.5h_1_s6.h5")
    # with h5py.File(fn) as f:
    #     lbl = f['lbl_nuc'][...]
    #     img = f['ch_03/0'][...]

    viewer = napari.Viewer()
    viewer.add_image(name='img', data=img)
    labels_layer = viewer.add_labels(name='lbl', data=lbl)

    widget = AnnotatorWidget(labels_layer, viewer, n_classes=3,
                             class_names=('M before division', 'M after division', 'S puncta', 'S high'))

    viewer.show(block=True)


class AnnotatorWidget:
    """Widget to assign labels in a labelimage to a class."""

    #TODO: Don't pass viewer.
    def __init__(self,
                 label_layer: Labels,
                 viewer: napari.Viewer,
                 n_classes: int = 2,
                 class_names: Optional[Sequence[str]] = None,
                 ):
        self.label_layer = label_layer
        self.label_layer.editable = False
        self.viewer = viewer
        self.cmap = ListedColormap([[0.0, 0.0, 0.0, 0.0]] + list(matplotlib.cm.get_cmap('Set1').colors))

        if class_names is None:
            self.n_classes = n_classes
            self.class_names = [f'Class {i + 1}' for i in range(n_classes)]
        if class_names is not None:
            if len(class_names) != n_classes and n_classes is not None:
                napari_info(f"Value provided for `n_classes` ({n_classes}) does not match the length of \
                            `class_names` ({len(class_names)}). Setting n_classes to {len(class_names)}")
            self.n_classes = len(class_names)
            self.class_names = list(class_names)

        self.annotations = pd.Series(index=np.unique(label_layer.data)[1:], name='annotations', dtype=pd.Int64Dtype)
        self.annotations.index.name = 'Label'


        # Create annotation layer (overwriting existing ones).
        if "annotation" in viewer.layers:
            viewer.layers.remove("annotation")
        self.annotation_layer = viewer.add_labels(
            label_layer.data, name="annotation", opacity=1.0, scale=label_layer.scale, translate=label_layer.translate,
        )
        self.annotation_layer.color[None] = np.array([0, 0, 0, 0.001], dtype=np.float32)
        self.annotation_layer.color_mode = 'direct'
        self.annotation_layer.editable = False

        widget = self.create_annotator_widget()
        viewer.window.add_dock_widget(widget, area="right", name=f"Annotator: {self.name}")

    def create_annotator_widget(self):  # pylint: disable-msg=R0915
        """
        Creates the annotator widget to choose a current class and export annotations.

        Parameters
        ----------
        label_layer: napari.layers.Labels
            The napari label layer on which objects shall be classified
        """
        # TODO: Ability dynamically change the names (and number) of classes (instead of having to provide
        #  them at initialization)?
        choices = ["Deselect"] + self.class_names
        selector = widgets.RadioButtons(
            choices=choices, label="Selection Class:", value=choices[1]
        )
        export_path = widgets.FileEdit(
            value=Path.cwd() / f"{self.name}_annotations.csv",
            label="Export Name:",
            mode="w",
            filter="*.csv",
        )
        export_button = widgets.PushButton(value=True, text="Export Annotations")
        container = widgets.Container(
            widgets=[
                selector,
                export_path,
                export_button,
            ]
        )

        @self.label_layer.mouse_drag_callbacks.append
        def toggle_label(_, event):  # pylint: disable-msg=W0613
            """
            Handles user annotations by setting the corresponding classifier
            variables and changing the
            """
            self.annotation_layer.visible = True
            # Need to scale position that event.position returns by the
            # label_layer scale.
            # If scale is (1, 1, 1), nothing changes
            # If scale is anything else, this makes the click still match the
            # correct label
            scaled_position = tuple(
                pos / scale for pos, scale in zip(event.position, self.label_layer.scale)
            )
            label = self.label_layer.get_value(scaled_position)
            if selector.value is None:
                napari_info(
                    "No class is selected. Select a class in the classifier widget."
                )
                return
            # Check if background or foreground was clicked. If background was
            # clicked, do nothing (background can't be assigned a class)
            if label == 0 or label is None:
                napari_info("No label clicked.")
                return
            self.annotations[label] = choices.index(selector.value)
            self.update_annotation_colormap(label, choices.index(selector.value))

        @selector.changed.connect
        def change_choice():
            self.annotation_layer.visible = True
            self.viewer.layers.selection.clear()
            # This doesn't work during testing
            try:
                self.viewer.layers.selection.add(self.label_layer)
            except ValueError:
                pass

        @export_button.changed.connect
        def export_annotations():
            if not str(export_path.value).endswith(".csv"):
                warnings.warn(
                    "The export path does not lead to a .csv file. This "
                    "export function will export in .csv format anyway"
                )

            # Check if file already exists
            if os.path.exists(Path(export_path.value)):
                msg_box = QMessageBox()
                msg_box.setText(
                    f"A csv export with the name {Path(export_path.value).name}"
                    " already exists. This will overwrite it."
                )
                msg_box.setWindowTitle("Overwrite Export?")
                msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                answer = msg_box.exec()
                if answer == QMessageBox.Cancel:
                    return
            napari_info("Exporting classifier results")

            self.annotations.to_csv(export_path.value)

        @self.label_layer.bind_key("b", overwrite=True)
        def toggle_selection_layer_visibility(layer):  # pylint: disable-msg=W0613
            self.annotation_layer.visible = not self.annotation_layer.visible

        @self.label_layer.bind_key("v", overwrite=True)
        def toggle_label_layer_visibility(layer):
            # Only set opacity to 0. Otherwise layer is not clickable anymore.
            if self.label_layer.opacity > 0:
                self.label_layer.opacity = 0.0
            else:
                self.label_layer.opacity = 0.8

        def set_class_n(event, n):
            selector.value = choices[n]
            change_choice()

        # keybindings for the available classes (0 = deselect)
        for i in range(self.n_classes + 1):
            set_class = partial(set_class_n, n=i)
            self.label_layer.bind_key(str(i), set_class)

        return container

    def update_annotation_colormap(self, label, new_class):
        """
        Updates the label colormap and sends the updated colormap to the label
        layer

        Parameters
        ----------
        label: int
            The label value that is being updated
        new_class: int
            The new class annotation that is being set
        """
        if label == 0 or label is None:
            return
        if new_class == 0:
            if label in self.annotation_layer.color.keys():
                self.annotation_layer.color.pop(label)
        else:
            self.annotation_layer.color[label] = self.cmap(new_class)
        self.annotation_layer.color_mode = 'direct'


if __name__ == '__main__':
    main()
