"""Annotator container widget for napari"""

import warnings
from collections.abc import Callable, Sequence
from enum import Enum
from functools import partial
from pathlib import Path

# pylint: disable=R0801
import napari
import napari.layers
import napari.viewer
import numpy as np
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    Container,
    FileEdit,
    Label,
    LineEdit,
    PushButton,
)

# pylint: disable=R0801
from napari_feature_classifier.utils import (
    add_annotation_names,
    get_colormap,
    get_selected_or_valid_label_layer,
    get_valid_label_layers,
    napari_info,
    overwrite_check_passed,
    reset_display_colormaps,
)


def get_class_selection(
    n_classes: int | None = None, class_names: Sequence[str] | None = None
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
            f"Setting n_classes to {len(class_names)}.",
            stacklevel=2,
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


class ClassSelectorRow(Container):
    """
    One row in `ClassSelectorPanel`: radio toggle, editable name, color swatch, count.

    Parameters
    ----------
    class_index : int
        1-based class index (matches the numeric annotation value).
    class_name : str
        Initial display name for this class.
    color : tuple[float, float, float, float]
        RGBA float tuple for the initial swatch color.
    on_select : Callable[[int], None]
        Called with `class_index` when the radio button is toggled on.
    on_color_changed : Callable[[int, tuple], None]
        Called with `(class_index, rgba)` when the user picks a new color.
    on_name_changed : Callable[[int, str], None]
        Called with `(class_index, new_name)` when the name edit changes.
    """

    def __init__(
        self,
        class_index: int,
        class_name: str,
        color: tuple[float, float, float, float],
        on_select: Callable[[int], None],
        on_color_changed: Callable[[int, tuple], None],
        on_name_changed: Callable[[int, str], None],
    ):
        self._class_index = class_index
        self._color = color
        self._on_select = on_select
        self._on_color_changed = on_color_changed
        self._on_name_changed = on_name_changed

        self._radio = CheckBox(value=False, label="")
        self._name_edit = LineEdit(value=class_name)
        self._color_btn = PushButton(text="")
        self._count_label = Label(value="0")

        super().__init__(
            widgets=[self._radio, self._name_edit, self._color_btn, self._count_label],
            layout="horizontal",
            labels=False,
        )
        # Uniform layout spacing so color swatches line up across rows
        self.native.layout().setContentsMargins(2, 1, 2, 1)
        self.native.layout().setSpacing(4)
        # Fixed-width count label so it doesn't shift the color swatch
        self._count_label.native.setFixedWidth(32)
        self._apply_color_style(color)
        self._radio.changed.connect(self._on_radio_changed)
        self._name_edit.changed.connect(
            lambda val: self._on_name_changed(self._class_index, val)
        )
        self._color_btn.clicked.connect(self._open_color_dialog)

    def _on_radio_changed(self, value: bool) -> None:
        if value:
            self._on_select(self._class_index)

    def _open_color_dialog(self) -> None:
        from qtpy.QtWidgets import QColorDialog

        r, g, b, a = (int(c * 255) for c in self._color)
        initial = __import__("qtpy.QtGui", fromlist=["QColor"]).QColor(r, g, b, a)
        color = QColorDialog.getColor(initial, options=QColorDialog.ShowAlphaChannel)
        if color.isValid():
            rgba = (
                color.redF(),
                color.greenF(),
                color.blueF(),
                color.alphaF(),
            )
            self.update_color(rgba)
            self._on_color_changed(self._class_index, rgba)

    def _apply_color_style(self, color: tuple[float, float, float, float]) -> None:
        r, g, b, a = (int(c * 255) for c in color)
        self._color_btn.native.setStyleSheet(
            f"background-color: rgba({r},{g},{b},{a}); min-width: 20px; max-width: 20px;"
        )

    def select(self) -> None:
        """Mark this row as selected without re-firing on_select."""
        self._radio.changed.disconnect(self._on_radio_changed)
        self._radio.value = True
        self._radio.changed.connect(self._on_radio_changed)

    def deselect(self) -> None:
        """Mark this row as deselected without re-firing on_select."""
        self._radio.changed.disconnect(self._on_radio_changed)
        self._radio.value = False
        self._radio.changed.connect(self._on_radio_changed)

    def update_count(self, n: int) -> None:
        self._count_label.value = str(n)

    def update_color(self, color: tuple[float, float, float, float]) -> None:
        self._color = color
        self._apply_color_style(color)

    @property
    def class_name(self) -> str:
        return self._name_edit.value


class ClassSelectorPanel(Container):
    """
    Vertical panel of `ClassSelectorRow` widgets — one per class.

    Replaces the magicgui `RadioButtons` widget in `LabelAnnotator`. Provides the
    same `.value` property interface (returns an Enum member) so that `toggle_label()`
    can still use `.value.value` to get the numeric annotation value.

    Parameters
    ----------
    ClassSelection : Enum
        The class selection enum (including NoClass at index 0).
    class_colors : dict[int, tuple]
        Per-class colors keyed by 1-based class index. Missing entries fall back
        to the Set1 colormap.
    on_names_changed : Callable[[list[str]], None]
        Fired when any class name is edited. Receives the full new name list.
    on_colors_changed : Callable[[int, tuple], None]
        Fired when a color swatch is changed. Receives (class_index, rgba).
    """

    def __init__(
        self,
        ClassSelection,  # noqa: N803
        class_colors: dict[int, tuple[float, float, float, float]] | None = None,
        on_names_changed: Callable[[list[str]], None] | None = None,
        on_colors_changed: Callable[[int, tuple], None] | None = None,
    ):
        self.ClassSelection = ClassSelection  # pylint: disable=C0103
        self._class_colors = class_colors or {}
        self._on_names_changed = on_names_changed or (lambda names: None)
        self._on_colors_changed = on_colors_changed or (lambda idx, rgba: None)
        self._selected_n = 1  # default: first real class (index 1 in __members__)
        self._cmap = get_colormap()

        # Build one row per class (skip NoClass at index 0)
        members = list(ClassSelection.__members__.keys())  # [NoClass, Class_1, ...]
        self._rows: list[ClassSelectorRow] = []
        for i, name in enumerate(members[1:], start=1):
            color = self._resolve_color(i)
            row = ClassSelectorRow(
                class_index=i,
                class_name=name,
                color=color,
                on_select=self._on_row_selected,
                on_color_changed=self._on_row_color_changed,
                on_name_changed=self._on_row_name_changed,
            )
            self._rows.append(row)

        self._no_class_btn = PushButton(text="No Class")
        super().__init__(widgets=[self._no_class_btn, *self._rows], labels=False)
        self._no_class_btn.clicked.connect(lambda: self.set_selected(0))
        # Select the first class by default
        if self._rows:
            self._rows[0].select()

    def _resolve_color(self, class_index: int) -> tuple[float, float, float, float]:
        """Return the stored color for class_index, falling back to Set1."""
        if class_index in self._class_colors:
            return self._class_colors[class_index]
        # Set1 fallback: normalize class_index into [0,1] for the colormap
        return tuple(self._cmap(class_index / len(self._cmap.colors)))

    def _on_row_selected(self, class_index: int) -> None:
        self._selected_n = class_index
        for row in self._rows:
            if row._class_index != class_index:
                row.deselect()

    def _on_row_color_changed(
        self, class_index: int, rgba: tuple[float, float, float, float]
    ) -> None:
        self._class_colors[class_index] = rgba
        self._on_colors_changed(class_index, rgba)

    def _on_row_name_changed(self, class_index: int, new_name: str) -> None:
        # Rebuild ClassSelection Enum with updated names
        new_names = [row.class_name for row in self._rows]
        # Apply the edit (row's LineEdit already has the new value)
        new_names[class_index - 1] = new_name
        try:
            self.ClassSelection = get_class_selection(class_names=new_names)
        except AssertionError:
            return  # Duplicate name — ignore until unique
        self._on_names_changed(new_names)

    @property
    def value(self):
        """Return the currently selected ClassSelection Enum member."""
        members = list(self.ClassSelection.__members__.keys())
        return self.ClassSelection[members[self._selected_n]]

    def set_selected(self, n: int) -> None:
        """
        Select class by position in ClassSelection.__members__ (0 = NoClass).
        n=0 deselects all rows; n=1..N selects the corresponding row.
        """
        self._selected_n = n
        for row in self._rows:
            if n > 0 and row._class_index == n:
                row.select()
            else:
                row.deselect()

    def update_counts(self, counts: dict[str, int]) -> None:
        """Update the count label on each row."""
        for row in self._rows:
            name = row.class_name
            row.update_count(counts.get(name, 0))

    def get_color(self, class_index: int) -> tuple[float, float, float, float]:
        """Return the current color for a 1-based class index."""
        if 1 <= class_index <= len(self._rows):
            return self._rows[class_index - 1]._color
        return (0.0, 0.0, 0.0, 0.0)


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
        ClassSelection=None,
        annotation_callbacks: list[Callable] | None = None,
        class_colors: dict[int, tuple[float, float, float, float]] | None = None,
        on_names_changed: Callable[[list[str]], None] | None = None,
        on_colors_changed: Callable[[int, tuple], None] | None = None,
    ):
        if ClassSelection is None:
            ClassSelection = get_class_selection(n_classes=4)
        self._viewer = viewer
        self._annotation_callbacks: list[Callable] = annotation_callbacks or []
        self._label_column = "label"

        self._last_selected_label_layer = get_selected_or_valid_label_layer(
            viewer=self._viewer
        )

        self.last_selected_layer_label = Label(
            label="Last selected label layer:", value=self._last_selected_label_layer
        )

        # Handle existing predictions layer
        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Labels) and layer.name == "Annotations":
                self._viewer.layers.remove(layer)
        self.add_annotations_layer()

        # Class selection panel (replaces RadioButtons)
        self.ClassSelection = ClassSelection  # pylint: disable=C0103
        self.nb_classes = len(self.ClassSelection) - 1
        self.cmap = get_colormap()
        self._class_selector = ClassSelectorPanel(
            ClassSelection=ClassSelection,
            class_colors=class_colors,
            on_names_changed=self._on_class_names_changed_internal(on_names_changed),
            on_colors_changed=self._on_class_colors_changed_internal(on_colors_changed),
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
        # Connect to label layer change, potentially call init
        self._viewer.layers.selection.events.changed.connect(self.selection_changed)

    def _on_class_names_changed_internal(
        self, external_cb: Callable[[list[str]], None] | None
    ) -> Callable[[list[str]], None]:
        """Return a callback that syncs ClassSelection on the annotator then calls external_cb."""

        def _cb(new_names: list[str]) -> None:
            self.ClassSelection = self._class_selector.ClassSelection
            if external_cb:
                external_cb(new_names)

        return _cb

    def _on_class_colors_changed_internal(
        self, external_cb: Callable[[int, tuple], None] | None
    ) -> Callable[[int, tuple], None]:
        """Return a callback that updates the annotation colormap then calls external_cb."""

        def _cb(class_index: int, rgba: tuple) -> None:
            # Re-render the annotations layer with the new color
            reset_display_colormaps(
                self._last_selected_label_layer,
                feature_col="annotations",
                display_layer=self._annotations_layer,
                label_column=self._label_column,
                color_resolver=self._class_selector.get_color,
            )
            if external_cb:
                external_cb(class_index, rgba)

        return _cb

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

    def add_annotations_layer(self):
        self._annotations_layer = self._viewer.add_labels(
            self._last_selected_label_layer.data,
            scale=self._last_selected_label_layer.scale,
            name="Annotations",
            translate=self._last_selected_label_layer.translate,
        )
        self._annotations_layer.editable = False
        # Set the label selection to a valid label layer
        self._viewer.layers.selection.active = self._last_selected_label_layer

    def toggle_label(self, labels_layer, event):
        """
        Callback for when a label is clicked. It then updates the color of that
        label in the annotation layer.
        """
        # If the annotations layer is missing, add it back
        if "Annotations" not in [x.name for x in self._viewer.layers]:
            self.add_annotations_layer()

        scaled_position = self.get_scaled_position(
            event.position, labels_layer.translate, labels_layer.scale
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
        for cb in self._annotation_callbacks:
            cb()

    @staticmethod
    def get_scaled_position(
        position: tuple, translate: np.array, scale: np.array
    ) -> tuple:
        """
        Get the position of a click after translation & scaling


        Position values in napari can have different shapes than the layer
        translate & scale data (e.g. 3D position for 2D layer). This function
        handles that edge-case
        """
        if len(position) == 3 and len(scale) == 2:
            position = position[1:]
        elif len(position) != len(scale):
            raise NotImplementedError(
                "Detecting annotation positions isn't implemented for "
                f"positions like {position} of length {len(position)} and "
                f"scales like {scale} of length {len(scale)}"
            )
        return tuple(
            (pos - trans) / scale
            for pos, trans, scale in zip(position, translate, scale, strict=False)
        )

    def set_class_n(self, event, n: int):  # pylint: disable=C0103
        self._class_selector.set_selected(n)

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

        self._annotations_layer.data = label_layer.data
        self._annotations_layer.scale = label_layer.scale
        self._annotations_layer.translate = label_layer.translate

        reset_display_colormaps(
            label_layer,
            feature_col="annotations",
            display_layer=self._annotations_layer,
            label_column=self._label_column,
            color_resolver=self._class_selector.get_color,
        )
        if self.toggle_label not in label_layer.mouse_drag_callbacks:
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

        napari does not have a direct API to only update a single color —
        it always validates & updates the whole colormap, so this scales
        with the number of unique labels.
        See https://github.com/napari/napari/issues/6732
        """
        from napari.utils.colormaps import DirectLabelColormap

        annotation_val = label_layer.features.loc[
            label_layer.features[self._label_column] == label,
            "annotations",
        ].iloc[0]
        import math as _math

        if isinstance(annotation_val, float) and _math.isnan(annotation_val):
            color = (0.0, 0.0, 0.0, 0.0)
        else:
            color = self._class_selector.get_color(int(annotation_val))
        colordict = self._annotations_layer.colormap.color_dict
        colordict[label] = color
        self._annotations_layer.colormap = DirectLabelColormap(color_dict=colordict)
        self._annotations_layer.opacity = 1.0

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
