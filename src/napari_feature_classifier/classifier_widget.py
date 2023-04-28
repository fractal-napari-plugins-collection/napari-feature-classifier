from pathlib import Path

from typing import Optional, Sequence, cast

import imageio
import napari
import napari.layers
import napari.viewer
import numpy as np
import pandas as pd
from magicgui.widgets import (
    Container,
    Label,
    FileEdit,
    LineEdit,
    PushButton,
    Select,
)
from napari.utils.notifications import show_info


from feature_loader_widget import LoadFeaturesContainer, make_features
from napari_feature_classifier.annotator_init_widget import LabelAnnotatorTextSelector
from napari_feature_classifier.annotator_widget import (
    LabelAnnotator,
    get_class_selection,
)
from napari_feature_classifier.classifier_new import Classifier
from napari_feature_classifier.utils import (
    get_colormap,
    reset_display_colormaps,
    get_valid_label_layers,
    get_selected_or_valid_label_layer,
)


def main():
    lbls = imageio.v2.imread(Path("sample_data/test_labels.tif"))
    lbls2 = np.zeros_like(lbls)
    lbls2[:, 3:, 2:] = lbls[:, :-3, :-2]
    lbls2 = lbls2 * 20

    labels = np.unique(lbls)[1:]
    labels_2 = np.unique(lbls2)[1:]

    viewer = napari.Viewer()
    lbls_layer = viewer.add_labels(lbls)
    lbls_layer2 = viewer.add_labels(lbls2)

    lbls_layer.features = make_features(labels, roi_id="ROI1", n_features=6)
    lbls_layer2.features = make_features(labels_2, roi_id="ROI2", n_features=6)
    # classifier_widget = ClassifierWidget(viewer)
    # load_widget = LoadFeaturesContainer(lbls_layer2)

    # viewer.window.add_dock_widget(classifier_widget)
    # viewer.window.add_dock_widget(load_widget)
    viewer.show(block=True)


class ClassifierInitContainer(Container):
    def __init__(self, viewer: napari.viewer.Viewer):
        self._viewer = viewer
        self._name_edit = LineEdit(value="classifier", label="Classifier Name:")
        try:
            self._last_selected_label_layer = get_selected_or_valid_label_layer(
                viewer=self._viewer
            )
        except NotImplementedError:
            self._last_selected_label_layer = None
        # TODO: Make this label left-aligned, not centered
        self.last_selected_layer_label = Label(
            label="Selecting features from:", value=self._last_selected_label_layer
        )
        self._feature_combobox = Select(
            choices=self.get_feature_options(self._last_selected_label_layer),
            allow_multiple=True,
            label="Feature Selection:",
        )
        self._label = 1
        self._annotation_name_selector = LabelAnnotatorTextSelector()
        self._initialize_button = PushButton(text="Initialize")
        super().__init__(
            widgets=[
                self._name_edit,
                self.last_selected_layer_label,
                self._feature_combobox,
                self._annotation_name_selector,
                self._initialize_button,
            ]
        )
        self._viewer.layers.selection.events.changed.connect(
            self.update_layer_selection
        )

    def get_selected_features(self):
        return self._feature_combobox.value

    def get_feature_options(self, layer):
        # Get the feature options of the currently selected layer
        # Only works if a label layer is selected (we don't load features from other layers)
        if isinstance(layer, napari.layers.Labels):
            return list(layer.features.columns)
        else:
            return []

    def update_layer_selection(self):
        if isinstance(self._viewer.layers.selection.active, napari.layers.Labels):
            self._last_selected_label_layer = self._viewer.layers.selection.active
            self.last_selected_layer_label.value = self._last_selected_label_layer
            self._feature_combobox.choices = self.get_feature_options(
                self._last_selected_label_layer
            )
            self._feature_combobox._default_choices = self.get_feature_options(
                self._last_selected_label_layer
            )


class ClassifierRunContainer(Container):
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        classifier: Optional[Classifier] = None,
        class_names: Optional[list[str]] = None,
        feature_names: Optional[list[str]] = None,
    ):
        self._viewer = viewer
        self._last_selected_label_layer = get_selected_or_valid_label_layer(
            viewer=self._viewer
        )
        # Initialize the classifier
        if classifier:
            self._classifier = classifier
            self.class_names = self._classifier.get_class_names()
            self.feature_names = self._classifier.get_feature_names()
        else:
            if not class_names or not feature_names:
                raise ValueError(
                    "A classifier object or "
                    "class_names & feature_names "
                    "must be provided"
                )
            self._classifier = Classifier(
                feature_names=feature_names, class_names=class_names
            )
            self.class_names = class_names
            self.feature_names = feature_names

        self._label_column = "label"
        self._roi_id_colum = "roi_id"

        self._annotator = LabelAnnotator(
            self._viewer, get_class_selection(class_names=self.class_names)
        )

        self._prediction_layer = self._viewer.add_labels(
            self._last_selected_label_layer.data,
            scale=self._last_selected_label_layer.scale,
            name="Predictions",
        )

        # Set the label selection to a valid label layer => Running into proxy bug
        self._viewer.layers.selection.active = self._last_selected_label_layer

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
        self._viewer.layers.selection.events.changed.connect(self.selection_changed)
        self._init_prediction_layer(self._last_selected_label_layer)

    def run(self):
        self.add_features_to_classifier()
        self._classifier.train()  # Show performance of training
        self.make_predictions()

    def add_features_to_classifier(self):
        # Generate a dict of features: Key are roi_ids, values are dataframes from layer.features.
        dict_of_features = {}
        for layer in self._viewer.layers:
            if (
                isinstance(layer, napari.layers.Labels)
                and len(layer.features) > 0
                and "annotations" in layer.features.columns
            ):
                # TODO: Add extra checks that it contains valid features?
                if "roi_id" in layer.features.columns:
                    roi_ids = layer.features["roi_id"].unique()
                    if len(roi_ids) > 1:
                        raise NotImplementedError(
                            f"{layer=} contained no-unique roi_ids: {roi_ids}"
                        )
                    else:
                        roi_id = roi_ids[0]
                        dict_of_features[roi_id] = layer.features
                else:
                    # TODO: Consider label-layer hashing here instead of using the layer name as roi_id
                    dict_of_features[layer.name] = layer.features
        self._classifier.add_dict_of_features(dict_of_features)

    def make_predictions(self):
        # Get all the label layers that have fitting features
        relevant_label_layers = self.get_relevant_label_layers()

        # Get the features dataframes with the relevant features
        prediction_dfs = {}
        for label_layer in relevant_label_layers:
            roi_id = self.get_layer_roi_id(label_layer)
            if roi_id in prediction_dfs.keys():
                raise ValueError(
                    f"Duplicate roi_id {roi_id} found in {label_layer.name}. "
                    "It's already present as the roi_id of another label layer. "
                )
            prediction_dfs[roi_id] = self.get_relevant_features(label_layer.features)

        # Get the classifier predictions
        prediction_results_dict = self._classifier.predict_on_dict(prediction_dfs)

        # Append the predictions to each open label layer ("predict" column)
        for label_layer in relevant_label_layers:
            roi_id = self.get_layer_roi_id(label_layer)
            # Merge the predictions back into the layer.features dataframe
            # TODO: Check that this merge is robust, never drops rows etc.
            if "predict" in label_layer.features.columns:
                label_layer.features.drop(columns=["predict"], inplace=True)
            label_layer.features = label_layer.features.merge(
                prediction_results_dict[roi_id],
                left_on=[self._label_column, self._roi_id_colum],
                right_index=True,
                how="outer",
            )
        
        self._init_prediction_layer(self._last_selected_label_layer)

    def selection_changed(self):
        # Check if the selection change results in a valid label layer being
        # selected. If so, initialize the prediction layer for it.
        if self._viewer.layers.selection.active:
            if self._viewer.layers.selection.active in get_valid_label_layers(
                viewer=self._viewer
            ):
                self._last_selected_layer = self._viewer.layers.selection.active
                self._init_prediction_layer(self._viewer.layers.selection.active)

    def _init_prediction_layer(self, label_layer: napari.layers.Labels):
        # Check if the predict column already exists in the layer.features
        if "predict" not in label_layer.features:
            unique_labels = np.unique(label_layer.data)[1:]
            predict_df = pd.DataFrame(
                {self._label_column: unique_labels, "predict": np.NaN}
            )
            if self._label_column in label_layer.features.columns:
                label_layer.features = label_layer.features.merge(
                    predict_df, on=self._label_column, how="outer"
                )
            else:
                label_layer.features = pd.concat(
                    [label_layer.features, predict_df], axis=1
                )

        # Update the label data in the prediction layer
        self._prediction_layer.data = label_layer.data
        self._prediction_layer.scale = label_layer.scale

        # Update the colormap of the prediction layer
        reset_display_colormaps(
            label_layer,
            feature_col="predict",
            display_layer=self._prediction_layer,
            label_column=self._label_column,
            cmap=get_colormap(),
        )

    def get_relevant_label_layers(self):
        relevant_label_layers = []
        required_columns = [self._label_column, self._roi_id_colum]
        excluded_label_layers = ["Annotations", "Predictions"]
        for label_layer in self._viewer.layers:
            if (
                isinstance(label_layer, napari.layers.Labels)
                and label_layer.name not in excluded_label_layers
            ):
                if label_layer.features is not None:
                    if all(x in label_layer.features.columns for x in required_columns):
                        relevant_label_layers.append(label_layer)
        return relevant_label_layers

    def get_layer_roi_id(self, label_layer):
        roi_ids = label_layer.features[self._roi_id_colum].unique()
        if len(roi_ids) > 1:
            raise NotImplementedError(
                f"{label_layer=} contained no-unique roi_ids: {roi_ids}"
            )
        return roi_ids[0]

    def get_relevant_features(self, df, filter_annotations: bool = False):
        # Get the relevant features from the pandas table
        # Creates double-indexing with label & roi_id?
        # filter_annotations: Only return rows that contain annotations?
        if not filter_annotations:
            df_relevant = df[
                [*self.feature_names, self._label_column, self._roi_id_colum]
            ]
            df_relevant.set_index(
                [self._label_column, self._roi_id_colum], inplace=True
            )
        else:
            df_relevant = df.loc[
                df["annotations"].notna(),
                [
                    *self.feature_names,
                    self._label_column,
                    self._roi_id_colum,
                    "annotations",
                ],
            ]
            df_relevant.set_index(
                [self._label_column, self._roi_id_colum], inplace=True
            )
        return df_relevant

    def save(self):
        # FIXME: Add options to define output_path & use that
        output_path = Path("sample_data/test_saving.clf")
        self._classifier.save(output_path)


class LoadClassifierContainer(Container):
    def __init__(self, viewer: napari.viewer.Viewer):
        self._viewer = viewer
        self._clf_destination = FileEdit(mode="r", filter="*.clf")
        self._load_button = PushButton(label="Load Classifier")
        super().__init__(widgets=[self._clf_destination, self._load_button])
        self._load_button.clicked.connect(self.load)

    def load(self):
        show_info("loading classifier")
        # FIXME: Load the actual classifier & pass it as an input
        # No more tmp class_names & feature_names
        class_names_tmp = ["Class 1", "Class 2", "Class 3"]
        feature_names_tmp = ["feature_1", "feature_2", "feature_3"]
        classifier = Classifier(feature_names_tmp, class_names_tmp)

        self._run_container = ClassifierRunContainer(self._viewer, classifier)
        self.clear()
        self.append(self._run_container)


class ClassifierWidget(Container):
    def __init__(self, viewer: napari.viewer.Viewer):
        self._viewer = viewer

        self._init_container = None
        self._run_container = None

        super().__init__(widgets=[])

        self.initialize_init_widget()

    def initialize_init_widget(self):
        self._init_container = ClassifierInitContainer(self._viewer)
        self.append(self._init_container)
        self._init_container._initialize_button.clicked.connect(
            self.initialize_run_widget
        )

    def initialize_run_widget(self):
        class_names = self._init_container._annotation_name_selector.get_class_names()
        feature_names = self._init_container.get_selected_features()
        if not feature_names:
            show_info("No features selected")
            return
        else:
            self._run_container = ClassifierRunContainer(
                self._viewer, class_names=class_names, feature_names=feature_names
            )
            self.clear()
            self.append(self._run_container)


if __name__ == "__main__":
    main()
