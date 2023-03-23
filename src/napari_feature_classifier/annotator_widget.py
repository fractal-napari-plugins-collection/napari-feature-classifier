import math
import warnings
from enum import Enum
from functools import partial
from typing import Optional, Sequence, cast

# from matplotlib.cm import get_cmap
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
    create_widget,
)
from matplotlib.colors import ListedColormap
from napari.utils.notifications import show_info


def get_class_selection(
    n_classes: Optional[int] = None, class_names: Optional[Sequence[str]] = None
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


def _get_appropriate_label_layer_names(
    viewer: napari.viewer.Viewer,
) -> list[napari.layers.Labels]:
    # Select label layers that are not `Annotation` or `Prediction`.
    return [
        layer.name
        for layer in viewer.layers
        if isinstance(layer, napari.layers.Labels)
        and layer.name not in ["Annotation", "Prediction"]
    ]


def _get_appropriate_label_layer(widget: ComboBox, viewer: napari.viewer.Viewer):
    return viewer.layers[widget.value]

# TODO: Put _lbl_combo Combobox into a separate Container so it can be used in other widgets.
# TODO: Make sure `Annotations` and `Predictions` don't show up in the ComboBox.
# TODO: Think about what happens when the widget is closed.
# TODO: Make sure annotation layer can't be selected in the `self._lbl_combo`.
class LabelAnnotator(Container):
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        ClassSelection=get_class_selection(n_classes=4),
    ):
        self._viewer = viewer

        self._lbl_combo = cast(ComboBox, create_widget(annotation=napari.layers.Labels))
        # Faided attempt at manually creating the ComboBox without Annotation and Prediction.
        # self._lbl_combo = ComboBox(
        #     choices=partial(_get_appropriate_label_layer_names, viewer=self._viewer),
        #     bind=partial(_get_appropriate_label_layer, viewer=self._viewer),
        # )

        self._annotations_layer = self._viewer.add_labels(
            self._lbl_combo.value.data,
            scale=self._lbl_combo.value.scale,
            name="Annotations",
        )

        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                layer.editable = False

        # Class selection
        self.ClassSelection = ClassSelection
        self.nb_classes = len(self.ClassSelection) - 1
        self.cmap = self.get_colormap()
        self._class_selector = cast(
            RadioButtons,
            create_widget(
                value=ClassSelection[list(ClassSelection.__members__.keys())[1]],
                annotation=ClassSelection,
                widget_type=RadioButtons,
            ),
        )
        self._init_annotation(self._lbl_combo.value)
        self._save_destination = FileEdit(value=f"annotation.csv", mode="r")
        self._save_annotation = PushButton(label="Save Annotations")
        self._update_save_destination(self._lbl_combo.value)
        super().__init__(
            widgets=[
                self._lbl_combo,
                self._class_selector,
                self._save_destination,
                self._save_annotation,
            ]
        )
        self._lbl_combo.changed.connect(self._on_label_layer_changed)
        self._viewer.layers.selection.events.changed.connect(self._active_changed)
        self._save_annotation.clicked.connect(self._on_save_clicked)

    def _init_annotation(self, label_layer: napari.layers.Labels):
        if "annotations" not in label_layer.features:
            unique_labels = np.unique(label_layer.data)[1:]
            label_layer.features["annotations"] = pd.Series(
                [np.NaN] * len(unique_labels),
                index=unique_labels,
            )
        self._lbl_combo.value.opacity = 0.4
        self._annotations_layer.data = label_layer.data
        self._annotations_layer.scale = label_layer.scale
        self._select_layer(label_layer)
        self.reset_annotation_colormaps()
        self._update_annotation_layer_name(label_layer)

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
            if label == 0 or not label:
                show_info("No label clicked.")
                return

            label_layer.features["annotations"].loc[
                label
            ] = self._class_selector.value.value

            # Update only the single color value that changed
            self.update_single_color(label)

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

    def _update_annotation_layer_name(self, label_layer: napari.layers.Labels):
        self._annotations_layer.name = (
            "Annotations"  # f"'{label_layer.name}' annotations"
        )

    def _on_label_layer_changed(self, label_layer: napari.layers.Labels):
        self._init_annotation(label_layer)
        self._update_save_destination(label_layer)
        self._update_annotation_layer_name(label_layer)
        # set your internal annotation layer here.

    def get_colormap(self, matplotlib_colormap="Set1"):
        """
        Generates colormaps depending on the number of classes
        """
        new_colors = np.array(matplotlib.colormaps[matplotlib_colormap].colors).astype(
            np.float32
        )
        cmap_np = np.zeros(
            shape=(new_colors.shape[0] + 1, new_colors.shape[1] + 1), dtype=np.float32
        )
        cmap_np[1:, :-1] = new_colors
        cmap_np[1:, -1] = 1
        cmap = ListedColormap(cmap_np)
        return cmap

    def reset_annotation_colormaps(self):
        """
        Reset the colormap based on the annotations in
        label_layer.features['annotation'] and sends the updated colormap
        to the annotation label layer
        """
        colors = self.cmap(
            self._lbl_combo.value.features["annotations"] / len(self.cmap.colors)
        )
        colordict = dict(zip(self._lbl_combo.value.features.index, colors))
        self._annotations_layer.color = colordict
        self._annotations_layer.opacity = 1.0
        self._annotations_layer.color_mode = "direct"

    def update_single_color(self, label):
        color = self.cmap(
            self._lbl_combo.value.features["annotations"][label] / len(self.cmap.colors)
        )
        self._annotations_layer.color[label] = color
        self._annotations_layer.opacity = 1.0
        self._annotations_layer.color_mode = "direct"

    def _select_layer(self, label_layer: napari.layers.Labels):
        self._viewer.layers.selection.clear()
        self._viewer.layers.selection.add(label_layer)

    def _active_changed(self, event):
        current_layer_proxy = self._viewer.layers.selection.active
        if current_layer_proxy is None:
            return

        if (
            current_layer_proxy.__class__ == napari.layers.Labels
            and current_layer_proxy.name != "Annotations"
        ):
            self._lbl_combo.value = self._viewer.layers.selection.active
        else:
            return
        # self._lbl_combo.value = self._viewer.layers.selection._current

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
        show_info(f"Annotations were saved at {self._save_destination.value}")
