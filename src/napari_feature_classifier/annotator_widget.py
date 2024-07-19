"""Annotator container widget for napari"""
import warnings
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, cast

# pylint: disable=R0801
import napari
import napari.layers
import napari.viewer
import numpy as np
import pandas as pd
from magicgui.widgets import (
    Label,
    Container,
    FileEdit,
    PushButton,
    RadioButtons,
    create_widget,
)


# pylint: disable=R0801
from napari_feature_classifier.utils import (
    get_colormap,
    reset_display_colormaps,
    get_valid_label_layers,
    get_selected_or_valid_label_layer,
    napari_info,
    overwrite_check_passed,
    add_annotation_names,
)


def get_class_selection(
    n_classes: Optional[int] = None, class_names: Optional[Sequence[str]] = None
) -> Enum:
    """
    Create a class selection enum for the annotator widget.

    Parameters:
    -----------
    n_classes: int, optional
        Number of classes to create the enum for. If not provided, the length of
        `class_names` is used.
    class_names: Sequence[str], optional
        List of class names to create the enum for. If not provided, the enum
        values will be `Class_1`, `Class_2`, etc., based on n_classes
    """
    if n_classes is None and class_names is None:
        raise ValueError("Provide either `n_classes` or a list of `class_names`")
    if class_names is None:
        class_names = [f"Class_{i+1}" for i in range(n_classes)]
    if n_classes is None:
        n_classes = len(class_names)
    if n_classes != len(class_names):
        warnings.warn(
            f"Value provided for `n_classes` ({n_classes}) does not match "
            f"the length of `class_names` ({len(class_names)}). "
            f"Setting n_classes to {len(class_names)}."
        )
    assert len(class_names) == len(
        set(class_names)
    ), f"{class_names=} contains duplicate entries!"

    # Setting NoClass to -1.0 => Deselection that is sent to the classifier
    # pylint: disable=C0103
    ClassSelection = Enum(
        "ClassSelection",
        {"NoClass": -1.0, **{c: i + 1 for i, c in enumerate(class_names)}},
    )
    return ClassSelection


# pylint: disable=R0902
class LabelAnnotator(Container):
    """
    The `LabelAnnotator` widget manages the annotation of a label layer by
    monitoring clicks on the selected label layer, adding annotations to the
    layer.features df and coloring an annotation layer accordingly.

    Paramters
    ---------
    viewer: napari.Viewer
        The current napari.Viewer instance
    ClassSelection: Enum
        The class selection to use for the annotation. Defaults to a 4 class selection.

    Attributes
    ----------
    viewer: napari.Viewer
        The current napari.Viewer instance
    _label_column: str
        The column name of the label column in the layer.features dataframe,
        hard-coded to "label"
    _last_selected_label_layer: napari.layers.Labels
        The last selected valid label layer
    last_selected_layer_label: magicgui.widgets.Label
        The Label widget for displaying the last selected label layer
    _annotations_layer: napari.layers.Labels
        The layer to on which annotations are displayed. This layer is not
        editable by the user.
    ClassSelection: Enum
        The class selection to use for the annotation.
    nb_classes: int
        The number of classes in the class selection (not counting deselection)
    cmap: matplotlib.colors.Colormap
        The colormap to use for the annotation layer
    _class_selector: magicgui.widgets.RadioButtons
        The RadioButtons widget for selecting the class to annotate.
        Can also be controlled via the number keys.
    """

    # TODO: Do we need to keep the annotation layer on top when new
    # annotations are made?
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        ClassSelection=get_class_selection(n_classes=4),
    ):
        self._viewer = viewer
        self._label_column = "label"

        self._last_selected_label_layer = get_selected_or_valid_label_layer(
            viewer=self._viewer
        )

        self.last_selected_layer_label = Label(
            label="Last selected label layer:", value=self._last_selected_label_layer
        )

        # Handle existing predictions layer
        for layer in self._viewer.layers:
            if type(layer) == napari.layers.Labels and layer.name == "Annotations":
                self._viewer.layers.remove(layer)
        self._annotations_layer = self._viewer.add_labels(
            self._last_selected_label_layer.data,
            scale=self._last_selected_label_layer.scale,
            name="Annotations",
            translate=self._last_selected_label_layer.translate,
        )
        self._annotations_layer.editable = False

        # Set the label selection to a valid label layer => Running into proxy bug
        self._viewer.layers.selection.active = self._last_selected_label_layer

        # Class selection
        self.ClassSelection = ClassSelection  # pylint: disable=C0103
        self.nb_classes = len(self.ClassSelection) - 1
        self.cmap = get_colormap()
        self._class_selector = cast(
            RadioButtons,
            create_widget(
                label="Class Selection",
                value=ClassSelection[list(ClassSelection.__members__.keys())[1]],
                annotation=ClassSelection,
                widget_type=RadioButtons,
            ),
        )
        self._init_annotation(self._last_selected_label_layer)
        self._save_destination = FileEdit(
            label="Save Path", value="annotation.csv", mode="w"
        )
        self._save_annotation = PushButton(label="Save Annotations")
        self._update_save_destination(self._last_selected_label_layer)
        super().__init__(
            widgets=[
                self.last_selected_layer_label,
                self._class_selector,
                self._save_destination,
                self._save_annotation,
            ]
        )
        self._save_annotation.clicked.connect(self._on_save_clicked)
        # TODO: Connect to the user clicking save in the file dialog. I can
        # trigger an event that the user clicked the button to open the
        # file dialog (see below), but don't get the info whether the user
        # clicked confirm or cancel.
        # self._save_destination.choose_btn.changed.connect(self.user_accepted)
        # Connect to label layer change, potentially call init
        self._viewer.layers.selection.events.changed.connect(self.selection_changed)

    def selection_changed(self, event):
        """
        Callback for when the selection changes. If the selection change results
        in a valid label layer being selected, initialize the annotator for it.
        """
        # Check if the selection change results in a valid label layer being
        # selected. If so, initialize the annotator for it.
        if self._viewer.layers.selection.active:
            if self._viewer.layers.selection.active in get_valid_label_layers(
                viewer=self._viewer
            ):
                self._init_annotation(self._viewer.layers.selection.active)
                self._save_annotation.enabled = True
                self._save_destination.enabled = True
                self._class_selector.enabled = True
                self.last_selected_layer_label.value = (
                    self._viewer.layers.selection.active
                )
                self._last_selected_label_layer = self._viewer.layers.selection.active
                self._update_save_destination(self._last_selected_label_layer)
            else:
                self._save_annotation.enabled = False
                self._save_destination.enabled = False
                self._class_selector.enabled = False
        else:
            self._save_annotation.enabled = False
            self._save_destination.enabled = False
            self._class_selector.enabled = False

    def toggle_label(self, labels_layer, event):
        """
        Callback for when a label is clicked. It then updates the color of that
        label in the annotation layer.
        """
        # Need to translate & scale position that event.position returns by the
        # label_layer scale.
        # If scale is (1, 1, 1), nothing changes
        # If translate is (0, 0, 0)
        # If scale is anything else, this makes the click still match the
        # correct label
        # translate before scale
        scaled_position = tuple(
            (pos - trans) / scale
            for pos, trans, scale in zip(
                event.position, labels_layer.translate, labels_layer.scale
            )
        )
        label = labels_layer.get_value(scaled_position)
        if label == 0 or not label:
            napari_info(f"No label clicked on the {labels_layer} label layer.")
            return

        # Left click: add annotation
        if event.button == 1:
            labels_layer.features.loc[
                labels_layer.features[self._label_column] == label, "annotations"
            ] = self._class_selector.value.value
        # Right click: Remove annotation
        elif event.button == 2:
            labels_layer.features.loc[
                labels_layer.features[self._label_column] == label, "annotations"
            ] = np.NaN

        # Update only the single color value that changed
        self.update_single_color(labels_layer, label)

    def set_class_n(self, event, n: int):  # pylint: disable=C0103
        self._class_selector.value = self.ClassSelection[
            list(self.ClassSelection.__members__)[n]
        ]

    def _init_annotation(self, label_layer: napari.layers.Labels):
        """
        Initializes the annotation layer for the given label layer.
        """
        label_layer.editable = False
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

        # label_layer.opacity = 0.4
        self._annotations_layer.data = label_layer.data
        self._annotations_layer.scale = label_layer.scale
        self._annotations_layer.translate = label_layer.translate

        reset_display_colormaps(
            label_layer,
            feature_col="annotations",
            display_layer=self._annotations_layer,
            label_column=self._label_column,
            cmap=self.cmap,
        )
        label_layer.mouse_drag_callbacks.append(self.toggle_label)

        # keybindings for the available classes (0 = deselect)
        for i in range(len(self.ClassSelection)):
            set_class = partial(self.set_class_n, n=i)
            set_class.__name__ = f"set_class_{i}"
            label_layer.bind_key(str(i), set_class, overwrite=True)

    def _update_save_destination(self, label_layer: napari.layers.Labels):
        """
        Update the default save destination to the name of the label layer.
        If a base_path was already set, keep it on that base path.

        """
        base_path = Path(self._save_destination.value).parent
        self._save_destination.value = base_path / f"{label_layer.name}_annotation.csv"

    def update_single_color(self, label_layer, label):
        """
        Update the color of a single object in the annotations layer.
        """
        color = self.cmap(
            float(
                label_layer.features.loc[
                    label_layer.features[self._label_column] == label,
                    "annotations",
                ].iloc[0]
            )
            / len(self.cmap.colors)
        )
        self._annotations_layer.color[label] = color
        self._annotations_layer.opacity = 1.0
        self._annotations_layer.color_mode = "direct"

    def _on_save_clicked(self):
        """
        Save annotations to a csv file.
        """
        # Check whether annotations should be overwritten.
        if not overwrite_check_passed(
            file_path=self._save_destination.value, output_type="annotation export"
        ):
            return

        annotations = self._last_selected_label_layer.features.loc[
            :, [self._label_column, "annotations"]
        ]
        # pylint: disable=C0103
        df = add_annotation_names(
            df=pd.DataFrame(annotations), ClassSelection=self.ClassSelection
        )

        df.to_csv(self._save_destination.value)
        napari_info(f"Annotations were saved at {self._save_destination.value}")
