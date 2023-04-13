import math
import warnings
from enum import Enum
from functools import partial
from typing import Optional, Sequence, cast

# from matplotlib.cm import get_cmap
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
from napari.utils.notifications import show_info

from napari_feature_classifier.utils import get_colormap, reset_display_colormaps
from napari_feature_classifier.label_layer_selector import LabelLayerSelector


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


# TODO: Think about what happens when the widget is closed.
class LabelAnnotator(Container):
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        ClassSelection=get_class_selection(n_classes=4),
    ):
        self._viewer = viewer
        self._label_column = "label"

        self._lbl_combo = LabelLayerSelector(viewer=self._viewer)

        current_selection = self._viewer.layers.selection.active
        print(self._viewer.layers.selection.active)

        self._annotations_layer = self._viewer.add_labels(
            self._lbl_combo.get_selected_label_layer().data,
            scale=self._lbl_combo.get_selected_label_layer().scale,
            name="Annotations",
        )
        print(self._viewer.layers.selection.active)
        # self._viewer.layers.selection.add(current_selection)
        print(self._viewer.layers.selection.active)
        self._lbl_combo.value = current_selection.name
        print(self._viewer.layers.selection.active)


        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                layer.editable = False

        # Class selection
        self.ClassSelection = ClassSelection
        self.nb_classes = len(self.ClassSelection) - 1
        self.cmap = get_colormap()
        self._class_selector = cast(
            RadioButtons,
            create_widget(
                value=ClassSelection[list(ClassSelection.__members__.keys())[1]],
                annotation=ClassSelection,
                widget_type=RadioButtons,
            ),
        )
        self._init_annotation(self._lbl_combo.get_selected_label_layer())
        self._save_destination = FileEdit(value=f"annotation.csv", mode="r")
        self._save_annotation = PushButton(label="Save Annotations")
        self._update_save_destination(self._lbl_combo.get_selected_label_layer())
        super().__init__(
            widgets=[
                self._lbl_combo,
                self._class_selector,
                self._save_destination,
                self._save_annotation,
            ]
        )
        self._save_annotation.clicked.connect(self._on_save_clicked)
        self._lbl_combo.changed.connect(self._on_label_layer_changed)

    # @self._lbl_combo.value.mouse_drag_callbacks.append
    def toggle_label(self, labels_layer, event):  # pylint: disable-msg=W0613
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

        labels_layer.features.loc[
            labels_layer.features[self._label_column] == label, "annotations"
        ] = self._class_selector.value.value

        # Update only the single color value that changed
        self.update_single_color(label)

    def set_class_n(self, layer, n: int):
        self._class_selector.value = self.ClassSelection[
            list(self.ClassSelection.__members__)[n]
        ]

    def _init_annotation(self, label_layer: napari.layers.Labels):
        if "annotations" not in label_layer.features:
            unique_labels = np.unique(label_layer.data)[1:]
            annotation_df = pd.DataFrame(
                {self._label_column: unique_labels, "annotations": np.NaN}
            )
            if self._label_column in label_layer.features.columns:
                label_layer.features = label_layer.features.merge(
                    annotation_df, on=self._label_column, how="outer"
                )
            else:
                label_layer.features = pd.concat(
                    [label_layer.features, annotation_df], axis=1
                )

        label_layer.opacity = 0.4
        self._annotations_layer.data = label_layer.data
        self._annotations_layer.scale = label_layer.scale
        reset_display_colormaps(
            label_layer,
            feature_col="annotations",
            display_layer=self._annotations_layer,
            label_column=self._label_column,
            cmap=self.cmap,
        )
        label_layer.mouse_drag_callbacks.append(self.toggle_label)

        # # keybindings for the available classes (0 = deselect)
        for i in range(len(self.ClassSelection)):
            set_class = partial(self.set_class_n, n=i)
            set_class.__name__ = f"set_class_{i}"
            label_layer.bind_key(str(i), set_class, overwrite=True)
        #     set_class = partial(set_class_n, n=i)
        #     self.label_layer.bind_key(str(i), set_class)

    def _update_save_destination(self, label_layer: napari.layers.Labels):
        self._save_destination.value = f"annotation_{label_layer.name}.csv"

    def _on_label_layer_changed(self):
        label_layer = self._lbl_combo.get_selected_label_layer()
        self._init_annotation(label_layer)
        self._update_save_destination(label_layer)

    def update_single_color(self, label):
        current_label_layer = self._lbl_combo.get_selected_label_layer()
        color = self.cmap(
            float(
                current_label_layer.features.loc[
                    current_label_layer.features[self._label_column] == label,
                    "annotations",
                ]
            )
            / len(self.cmap.colors)
        )
        self._annotations_layer.color[label] = color
        self._annotations_layer.opacity = 1.0
        self._annotations_layer.color_mode = "direct"

    def _on_save_clicked(self):
        annotations = self._lbl_combo.get_selected_label_layer().features["annotations"]
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
