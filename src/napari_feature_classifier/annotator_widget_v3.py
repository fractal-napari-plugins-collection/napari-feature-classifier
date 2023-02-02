import math
import warnings
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, cast

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
    create_widget,
)
from matplotlib.colors import ListedColormap


def main():
    import imageio

    lbls = imageio.v2.imread(
        r"C:\Users\hessm\Documents\Programming\Python\fractal-napari-plugins\napari-feature-classifier\src\napari_feature_classifier\sample_data\test_labels.tif"
    )
    viewer = napari.Viewer()
    viewer.add_labels(lbls)
    viewer.add_labels(lbls, name="lbls2")
    widget = viewer.window.add_dock_widget(
        LabelAnnotator(
            viewer,
            get_class_selection(
                class_names=[
                    "early M",
                    "late M",
                    "early S",
                    "mid S",
                    "late S",
                ]
            ),
        )
    )
    viewer.show(block=True)


def get_class_selection(
    n_classes: Optional[int] = 4, class_names: Optional[Sequence[str]] = None
) -> Enum:
    if n_classes is None and class_names is None:
        raise ValueError("Provide either `n_classes` or a list of `class_names`")
    if class_names is None:
        class_names = [f"Class_{i+1}" for i in range(n_classes)]
    if n_classes is None:
        n_classes = len(class_names)
    if n_classes != len(class_names):
        warnings.warn(
            f"Value provided for `n_classes` ({n_classes}) does not match the length of `class_names` ({len(class_names)}). Setting n_classes to {len(class_names)}"
        )
    assert len(class_names) == len(
        set(class_names)
    ), f"{class_names=} contains duplicate entries!"

    ClassSelection = Enum(
        "ClassSelection",
        {"NoClass": np.nan, **{c: i + 1 for i, c in enumerate(class_names)}},
    )
    return ClassSelection


# TODO: Init widget for specifying class names & number.
# TODO: All label layers (at the creation of the widget) are set to ´editable = False´ to make sure there's no weird interactions with label-layer functionality. Should this apply
#      to all label layers that are created while the plug-in extists and be reset when closing?
# TODO: Make sure user feedback (like "no label clicked.") ends up in the right place (i. e. the viewer info-bar).
# TODO: Remove the _lbl_combo box and only rely on layer selection in the viewer. This could solve a second issue that changes of the label layer name don't end up in the _lbl_combo box.
class LabelAnnotator(Container):
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        ClassSelection=get_class_selection(n_classes=4),
    ):
        self._viewer = viewer

        self._lbl_combo = cast(ComboBox, create_widget(annotation=napari.layers.Labels))
        self._lbl_combo.changed.connect(self._on_label_layer_changed)

        self._annotations_layer = self._viewer.add_labels(
            self._lbl_combo.value.data,
            scale=self._lbl_combo.value.scale,
            name="Annotations",
        )

        for layer in self._viewer.layers:
            print(layer)
            if isinstance(layer, napari.layers.Labels):
                layer.editable = False

        # Class selection
        self.ClassSelection = ClassSelection
        self._class_selector = cast(
            RadioButtons,
            create_widget(
                value=self.ClassSelection[
                    list(self.ClassSelection.__members__.keys())[1]
                ],
                annotation=self.ClassSelection,
                widget_type=RadioButtons,
            ),
        )
        self._init_annotation(self._lbl_combo.value)
        self._viewer.layers.selection.events.changed.connect(self._active_changed)
        self._save_destination = FileEdit(value=f"annotation.csv", mode="r")
        self._save_annotation = PushButton(label="Save Annotations")
        self._save_annotation.clicked.connect(self._on_save_clicked)
        self._update_save_destination(self._lbl_combo.value)
        super().__init__(
            widgets=[
                self._lbl_combo,
                self._class_selector,
                self._save_destination,
                self._save_annotation,
            ]
        )

    # TODO: Can we do a better separation of concerns (i. e. initializing label_layer.features['annotation'] vs. setting up keybindings)?
    def _init_annotation(self, label_layer: napari.layers.Labels):
        self._select_layer(label_layer)
        # print(self._save_destination)

        if "annotations" in label_layer.features:
            self.reset_annotation_colormaps()
            return
        unique_labels = np.unique(label_layer.data)[1:]
        label_layer.features["annotations"] = pd.Series(
            [np.NaN] * len(unique_labels), index=unique_labels, dtype=int
        )
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
                print("No label clicked.")
                return

            label_layer.features["annotations"].loc[
                label
            ] = self._class_selector.value.value

            # FIXME: Joel => Don't reset the whole colormap, just update a single color
            # Waiting for update on the napari 0.4.17 issue that breaks this option
            self.reset_annotation_colormaps()

        def set_class_n(layer, n: int):
            self._class_selector.value = self.ClassSelection[
                list(self.ClassSelection.__members__)[n]
            ]

        # # keybindings for the available classes (0 = deselect)
        for i in range(len(self.ClassSelection)):
            set_class = partial(set_class_n, n=i)
            set_class.__name__ = f"set_class_{i}"
            self._lbl_combo.value.bind_key(str(i), set_class, overwrite=True)
        #     set_class = partial(set_class_n, n=i)
        #     self.label_layer.bind_key(str(i), set_class)

    def _update_save_destination(self, label_layer: napari.layers.Labels):
        self._save_destination.value = f"annotation_{label_layer.name}.csv"

    def _on_label_layer_changed(self, label_layer: napari.layers.Labels):
        print("Label layer changed", label_layer)
        self._init_annotation(label_layer)
        self._update_save_destination(label_layer)
        # set your internal annotation layer here.

    # FIXME: Currently only works for <4 classes. Add more colors and make sure the mapping logic works independently of the number of classes.
    # TODO: Let's increase the max number to 9 (=> keybinding) and check that max number classes is <=9
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
        colors = cmap(
            self._lbl_combo.value.features["annotations"] / 5
        )  # self.nb_classes
        colordict = dict(zip(self._lbl_combo.value.features.index, colors))
        self._annotations_layer.color = colordict
        self._annotations_layer.opacity = 1.0
        self._annotations_layer.color_mode = "direct"

    def _select_layer(self, label_layer: napari.layers.Labels):
        print("selecting layer...")
        self._viewer.layers.selection.clear()
        self._viewer.layers.selection.add(label_layer)

    def _active_changed(self, event):
        print("selection changed...")
        current_layer = self._viewer.layers.selection._current
        if (
            type(current_layer) is napari.layers.Labels
            and current_layer.name != "Annotations"
        ):
            self._lbl_combo.value = self._viewer.layers.selection._current
        else:
            return
        # self._lbl_combo.value = self._viewer.layers.selection._current

    # TODO: Add user feedback in the viewer after annotations are saved.
    def _on_save_clicked(self):
        annotations = self._lbl_combo.value.features["annotations"]
        df = pd.DataFrame(annotations)
        class_names = []
        for annotation in annotations:
            if math.isnan(annotation):
                class_names.append(self.ClassSelection(np.NaN).name)
            else:
                class_names.append(self.ClassSelection(annotation).name)

        df["annotation_names"] = class_names
        df.to_csv(self._save_destination.value)


if __name__ == "__main__":
    main()
