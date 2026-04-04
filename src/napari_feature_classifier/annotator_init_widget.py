"""Annotator init widget for napari"""

import napari
import napari.layers
import napari.viewer
from magicgui.widgets import Container, LineEdit, PushButton

from napari_feature_classifier.annotator_widget import (
    LabelAnnotator,
    get_class_selection,
)


class LabelAnnotatorTextSelector(Container):
    """
    The `LabelAnnotatorTextSelector` container is a helper container for the
    label annotator where the user can name the classes to be annotated.

    Starts with `default_n_classes` pre-filled boxes. Additional classes can
    be added one at a time via the "Add Class" button, up to MAX_CLASSES.

    Parameters
    ----------
    default_n_classes: int
        The number of pre-filled class boxes to show on init. Defaults to 2.
    """

    MAX_CLASSES: int = 9

    def __init__(self, default_n_classes=2):
        self._text_edits: list[LineEdit] = [
            LineEdit(value=f"Class_{i + 1}", nullable=True)
            for i in range(default_n_classes)
        ]
        self._add_button = PushButton(text="Add Class")
        super().__init__(widgets=[*self._text_edits, self._add_button], labels=False)
        self.native.layout().setContentsMargins(0, 0, 0, 0)
        self._add_button.clicked.connect(self._add_class)

    def _add_class(self) -> None:
        """Append a new empty LineEdit before the Add Class button, up to MAX_CLASSES."""
        if len(self._text_edits) >= self.MAX_CLASSES:
            return
        new_edit = LineEdit(nullable=True)
        self._text_edits.append(new_edit)
        # Insert before the Add Class button (last widget)
        self.insert(len(self) - 1, new_edit)
        if len(self._text_edits) >= self.MAX_CLASSES:
            self._add_button.enabled = False

    def get_class_names(self):
        class_names = [e.value for e in self._text_edits if e.value != ""]
        return class_names


class InitializeLabelAnnotatorWidget(Container):
    """
    The `InitializeLabelAnnotatorWidget` container is an entry point to start
    an annotator without a classifier.

    Paramters
    ---------
    viewer: napari.Viewer
        The current napari.Viewer instance
    default_n_classes: int
        The number of classes to display. Defaults to 2.
    """

    def __init__(self, viewer: napari.viewer.Viewer, default_n_classes=2):
        self.viewer = viewer
        self.label_class_container = LabelAnnotatorTextSelector(default_n_classes)
        self._init_button = PushButton(label="Initialize")
        super().__init__(widgets=[self.label_class_container, self._init_button])
        self._init_button.clicked.connect(self.initialize_annotator)

    def initialize_annotator(self):
        class_names = self.label_class_container.get_class_names()
        annotator = LabelAnnotator(
            self.viewer, get_class_selection(class_names=class_names)
        )
        self.clear()
        self.append(annotator)
