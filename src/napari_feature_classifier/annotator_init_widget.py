import napari
import napari.layers
import napari.viewer
from magicgui.widgets import Container, LineEdit, PushButton

from napari_feature_classifier.annotator_widget import (
    LabelAnnotator,
    get_class_selection,
)


class LabelAnnotatorTextSelector(Container):
    MAX_CLASSES: int = 9

    def __init__(self, default_n_classes=5):
        default_line_edits = [
            LineEdit(value=f"Class_{i + 1}", nullable=True)
            for i in range(default_n_classes)
        ]
        empty_line_edits = [
            LineEdit(nullable=True) for i in range(self.MAX_CLASSES - default_n_classes)
        ]

        self._text_edits = tuple([*default_line_edits, *empty_line_edits])

        super().__init__(widgets=[*self._text_edits])

    def get_class_names(self):
        class_names = [
            e.value for e in self._text_edits if e.value != ""
        ]
        return class_names


class InitializeLabelAnnotatorWidget(Container):
    def __init__(self, viewer: napari.viewer.Viewer, default_n_classes=5):
        self.viewer = viewer
        self.label_class_container = LabelAnnotatorTextSelector(default_n_classes)
        self._init_button = PushButton(label="Initialize")
        super().__init__(widgets=[self.label_class_container, self._init_button])
        self._init_button.clicked.connect(self.initialize_annotator)

    def initialize_annotator(self):
        class_names = self.label_class_container.get_class_names()
        self.viewer.window.add_dock_widget(
            LabelAnnotator(self.viewer, get_class_selection(class_names=class_names))
        )
        # This closes the initialization dockwidget
        self.viewer.window.remove_dock_widget(self.native)
