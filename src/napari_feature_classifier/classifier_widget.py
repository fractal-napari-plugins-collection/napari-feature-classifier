from pathlib import Path

import imageio
import napari
import napari.layers
import napari.viewer
import numpy as np
from magicgui.widgets import Container, FileEdit, LineEdit, PushButton, Select
from napari.utils.notifications import show_info

from feature_loader_widget import LoadFeaturesContainer, make_features
from napari_feature_classifier.annotator_init_widget import LabelAnnotatorTextSelector
from napari_feature_classifier.annotator_widget import (
    LabelAnnotator,
    get_class_selection,
)


def main():
    lbls = imageio.imread(Path("sample_data/test_labels.tif"))
    lbls2 = np.zeros_like(lbls)
    lbls2[:, 3:, 2:] = lbls[:, :-3, :-2]
    lbls2 = lbls2 * 20

    labels = np.unique(lbls)[1:]

    viewer = napari.Viewer()
    lbls_layer = viewer.add_labels(lbls)
    lbls_layer2 = viewer.add_labels(lbls2)

    lbls_layer.features = make_features(labels, n_features=6)
    classifier_widget = ClassifierWidget(viewer)
    load_widget = LoadFeaturesContainer(lbls_layer2)

    viewer.window.add_dock_widget(classifier_widget)
    viewer.window.add_dock_widget(load_widget)
    viewer.show(block=True)
    dir(lbls_layer.features)


class ClassifierInitContainer(Container):
    def __init__(self, feature_options: list[str]):
        self._name_edit = LineEdit(value="classifier", label="Classifier Name:")
        self._feature_combobox = Select(
            choices=feature_options, allow_multiple=True, label="Feature Selection:"
        )
        self._annotation_name_selector = LabelAnnotatorTextSelector()
        self._initialize_button = PushButton(text="Initialize")
        super().__init__(
            widgets=[
                self._name_edit,
                self._feature_combobox,
                self._annotation_name_selector,
                self._initialize_button,
            ]
        )


class ClassifierRunContainer(Container):
    def __init__(self, viewer: napari.viewer.Viewer, class_names: list[str]):
        # Optionally get a classifier object or initialize one
        self._annotator = LabelAnnotator(
            viewer, get_class_selection(class_names=class_names)
        )
        self._run_button = PushButton(text="Run Classifier")
        self._save_button = PushButton(text="Save Classifier")
        super().__init__(
            widgets=[
                self._annotator,
                self._run_button,
                self._save_button,
            ]
        )
        self._run_button.clicked.connect(self.run)
        self._save_button.clicked.connect(self.save)

    def run(self):
        # TODO:
        # 1. Scan all open label layers for annotation & features [ignore annotation layer and predict layer]
        # 2. Update classifier internal feature store
        # 3. Train the classifier
        # 4. Update the prediction layer (create if non-existent) [for one label image => which one]

        show_info("running classifier...")

    def save(self):
        show_info("saving classifier...")


class LoadClassifierContainer(Container):
    # TODO: Implement this. Separate container that leads to run
    # Path to the classifier
    pass


class ClassifierWidget(Container):
    def __init__(self, viewer: napari.viewer.Viewer):
        self._viewer = viewer

        self._init_container = None
        self._run_container = None

        super().__init__(widgets=[])

        self.initialize_init_widget()

    def initialize_init_widget(self):
        # Extract features for first label layer
        # TODO: Handle case where there's no layers.features in the first Labels layer.
        label_layer = [
            l for l in self._viewer.layers if isinstance(l, napari.layers.Labels)
        ][0]
        feature_names = list(label_layer.features.columns)
        self._init_container = ClassifierInitContainer(feature_names)
        self.append(self._init_container)
        self._init_container._initialize_button.clicked.connect(
            self.initialize_run_widget
        )

    def initialize_run_widget(self):
        class_names = self._init_container._annotation_name_selector.get_class_names()
        self._run_container = ClassifierRunContainer(self._viewer, class_names)
        self.clear()
        self.append(self._run_container)


if __name__ == "__main__":
    main()
