"""Classifier container widget for napari"""
import logging
import pickle

from pathlib import Path
from typing import Optional

import napari
import napari.layers
import napari.viewer
import numpy as np
import pandas as pd
from magicgui.widgets import (
    Container,
    Label,
    FileEdit,
    RadioButtons,
    PushButton,
    Select,
)

from napari_feature_classifier.annotator_init_widget import LabelAnnotatorTextSelector
from napari_feature_classifier.annotator_widget import (
    LabelAnnotator,
    get_class_selection,
)
from napari_feature_classifier.classifier import Classifier
from napari_feature_classifier.utils import (
    get_colormap,
    reset_display_colormaps,
    get_valid_label_layers,
    get_selected_or_valid_label_layer,
    napari_info,
    overwrite_check_passed,
    add_annotation_names,
    NapariHandler,
)


class ClassifierInitContainer(Container):
    """
    The ClassifierInitContainer presents all the options needed for
    initializing a `ClassifierRunContainer`. It's intended as a container
    that's used in other magicgui containers.

    It offers feature selection for the last label layer that was selected
    (or a valid label layer if none had been selected). Changing the selected
    label layer changes the feature selection that is offered.
    No action is bound to the _initialize_button => needs to be done in the
    parent container to bind the correct run action (e.g. start a
    `ClassifierRunContainer` with the correct parameters)

    Paramters
    ---------
    viewer: napari.Viewer
        The current napari.Viewer instance

    Attributes
    ----------
    viewer: napari.Viewer
        The current napari.Viewer instance
    _last_selected_label_layer: napari.layers.Labels
        The last selected label layer
    last_selected_layer_label: magicgui.widgets.Label
        The Label widget for displaying the last selected label layer
    _feature_combobox: magicgui.widgets.Select
        The Select widget for selecting the features to use for classification
    _annotation_name_selector: LabelAnnotatorTextSelector
        The LabelAnnotatorTextSelector widget for selecting the annotation names
    _initialize_button: magicgui.widgets.PushButton
        The PushButton widget for initializing the classifier
    """

    def __init__(self, viewer: napari.viewer.Viewer):
        self._viewer = viewer
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
        self._annotation_name_selector = LabelAnnotatorTextSelector()
        # pylint: disable=W0212
        self._initialize_button = PushButton(text="Initialize")
        super().__init__(
            widgets=[
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
        """
        Returns the currently selected features (0-n features)
        """
        return self._feature_combobox.value

    def get_class_names(self):
        """
        Returns the available class names of the classifier
        """
        return self._annotation_name_selector.get_class_names()

    def get_feature_options(self, layer):
        """
        Get the feature options of the currently selected layer

        Only works if a label layer is selected (we don't load features from
        other layers)
        """
        if isinstance(layer, napari.layers.Labels):
            return list(layer.features.columns)
        return []

    def update_layer_selection(self):
        """
        Update the layer selection and the feature options if the newly
        selected layer is a label layer
        """
        if isinstance(self._viewer.layers.selection.active, napari.layers.Labels):
            self._last_selected_label_layer = self._viewer.layers.selection.active
            self.last_selected_layer_label.value = self._last_selected_label_layer
            self._feature_combobox.choices = self.get_feature_options(
                self._last_selected_label_layer
            )
            # pylint: disable=W0212
            self._feature_combobox._default_choices = self.get_feature_options(
                self._last_selected_label_layer
            )


# pylint: disable=R0902
class ClassifierRunContainer(Container):
    """
    The `ClassifierRunContainer` manages all the options needed to do
    annotations, train a classifier and show classifier results.

    The `ClassifierRunContainer` can be initialized with either an existing
    classifier or with class_names + feature_names

    Paramters
    ---------
    viewer: napari.Viewer
        The current napari.Viewer instance
    classifier: Optional[Classifier]
        The container can be initialized with an existing classifier. If none
        is provided, a classifier is generated upon init.
    class_names: Optional[list[str]]
        The class names of the classifier. Needs to be provided if no
        classifier is provided.
    feature_names: Optional[list[str]]
        The feature names of the classifier. Needs to be provided if no
        classifier is provided.
    classifier_save_path: Optional[str]
        The path to save the classifier to. If none is provided, the
        classifier creates a default path in the current working directory.
    auto_save: Optional[bool]
        Whether the classifier can automatically save at the
        classifier_save_path upon run and save. If false, run does not
        trigger a save and save checks for overwrite conflicts.

    Attributes
    ----------
    viewer: napari.Viewer
        The current napari.Viewer instance
    _last_selected_label_layer: napari.layers.Labels
        The last selected label layer
    _classifier: Classifier
        The classifier object
    class_names: list[str]
        The class names of the classifier classes. Matching the first name to
        1, second to 2 etc.
    feature_names: list[str]
        The feature names of the classifier. These features are loaded from
        layer.features to train the classifier
    _label_column: str
        The column name of the label column in the layer.features dataframe,
        hard-coded to "label"
    _roi_id_column: str
        The column name of the roi_id column in the layer.features dataframe,
        hard-coded to "roi_id"
    _annotator: LabelAnnotator
        The LabelAnnotator container that manages annotations
    _prediction_layer: napari.layers.Labels
        The prediction layer that is generated by the classifier and which
        displays predictions made for the currently selected label layer.
    _run_button: magicgui.widgets.PushButton
        The PushButton widget for running the classifier
    _save_destination: magicgui.widgets.FileEdit
        The FileEdit widget for selecting the save destination of the classifier
    _save_button: magicgui.widgets.PushButton
        The PushButton widget for saving the classifier
    """

    # pylint: disable=R0913
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        classifier: Optional[Classifier] = None,
        class_names: Optional[list[str]] = None,
        feature_names: Optional[list[str]] = None,
        classifier_save_path: Optional[str] = None,
        auto_save: Optional[bool] = False,
    ):
        self._viewer = viewer
        self.auto_save = auto_save
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

        # Handle existing predictions layer
        for layer in self._viewer.layers:
            if type(layer) == napari.layers.Labels and layer.name == "Predictions":
                self._viewer.layers.remove(layer)
        self._prediction_layer = self._viewer.add_labels(
            self._last_selected_label_layer.data,
            scale=self._last_selected_label_layer.scale,
            name="Predictions",
            translate=self._last_selected_label_layer.translate,
        )

        # Set the label selection to a valid label layer => Running into proxy bug
        self._viewer.layers.selection.active = self._last_selected_label_layer

        self._run_button = PushButton(text="Run Classifier")
        self._save_destination = FileEdit(
            label="Classifier Save Path",
            value=f"{self._last_selected_label_layer}_classifier.clf",
            mode="w",
        )
        if classifier_save_path:
            self._save_destination.value = classifier_save_path
        self._save_button = PushButton(text="Save Classifier")

        # Export options
        self._export_destination = FileEdit(
            label="Prediction Export Path",
            value=f"{self._last_selected_label_layer}_predictions.csv",
            mode="w",
        )
        self._export_button = PushButton(text="Export Classifier Result")
        super().__init__(
            widgets=[
                self._annotator,
                self._save_destination,
                self._run_button,
                self._save_button,
                self._export_destination,
                self._export_button,
            ]
        )
        self._run_button.clicked.connect(self.run)
        self._save_button.clicked.connect(self.save)
        self._export_button.clicked.connect(self.export_results)
        self._viewer.layers.selection.events.changed.connect(self.selection_changed)
        self._init_prediction_layer(self._last_selected_label_layer)
        # Whenever the label layer is clicked, hide the prediction layer
        # (e.g. new annotations are made)
        self._last_selected_label_layer.mouse_drag_callbacks.append(
            self.hide_prediction_layer
        )

    def run(self):
        """
        Run method that adds features to the classifier, trains it, triggers
        predictions & saves the classifier
        """

        self.add_features_to_classifier()
        try:
            self._classifier.train()
        except ValueError as e:
            napari_info(
                "Training failed. A typical reason are not having "
                "enough annotations. \nThe error message was: "
                f"{e}"
            )
        else:
            self.make_predictions()
            self._prediction_layer.visible = True
            self.save()

    def add_features_to_classifier(self):
        """
        Generate a dict of features: Key are roi_ids, values are dataframes
        from layer.features.
        """
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

                    roi_id = roi_ids[0]
                    dict_of_features[roi_id] = layer.features
                else:
                    # TODO: Consider label-layer hashing here instead of
                    # using the layer name as roi_id
                    dict_of_features[layer.name] = layer.features
        self._classifier.add_dict_of_features(dict_of_features)

    def make_predictions(self):
        """
        Make predictions for all relevant label layers and add them to the
        layer.features `prediction` column of each layer
        """
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
            prediction_dfs[roi_id] = self.get_relevant_features(
                label_layer.features, set_index=False
            )

        # Get the classifier predictions
        prediction_results_dict = self._classifier.predict_on_dict(prediction_dfs)

        # Append the predictions to each open label layer ("prediction" column)
        for label_layer in relevant_label_layers:
            roi_id = self.get_layer_roi_id(label_layer)
            if "prediction" in label_layer.features.columns:
                label_layer.features.drop(columns=["prediction"], inplace=True)
            # Merge the predictions back into the layer.features dataframe
            label_layer.features = label_layer.features.merge(
                prediction_results_dict[roi_id],
                left_on=[self._roi_id_colum, self._label_column],
                right_index=True,
                how="outer",
            )

        self._init_prediction_layer(self._last_selected_label_layer)

    def selection_changed(self):
        """
        Check if the selection change results in a valid label layer being
        selected. If so, initialize the prediction layer for it.
        """
        if self._viewer.layers.selection.active:
            if self._viewer.layers.selection.active in get_valid_label_layers(
                viewer=self._viewer
            ):
                self._last_selected_label_layer = self._viewer.layers.selection.active
                self._init_prediction_layer(self._viewer.layers.selection.active)
                self._last_selected_label_layer.mouse_drag_callbacks.append(
                    self.hide_prediction_layer
                )
                self._update_export_destination(self._last_selected_label_layer)

    def _init_prediction_layer(self, label_layer: napari.layers.Labels):
        """
        Initialize the prediction layer and reset its data (to fit the input
        label_layer) and its colormap
        """
        # Check if the predict column already exists in the layer.features
        if "prediction" not in label_layer.features:
            unique_labels = np.unique(label_layer.data)[1:]
            predict_df = pd.DataFrame(
                {self._label_column: unique_labels, "prediction": np.NaN}
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
        self._prediction_layer.translate = label_layer.translate

        # Update the colormap of the prediction layer
        reset_display_colormaps(
            label_layer,
            feature_col="prediction",
            display_layer=self._prediction_layer,
            label_column=self._label_column,
            cmap=get_colormap(),
        )

    def hide_prediction_layer(self, labels_layer, event):
        """
        Hide the prediction layer
        """
        self._prediction_layer.visible = False

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

    # pylint: disable=C0103
    def get_relevant_features(
        self, df, filter_annotations: bool = False, set_index=False
    ):
        """
        Get the relevant features from the pandas table
        Can optionally create a double-indexing with label & roi_id
        filter_annotations: Only return rows that contain annotations?
        """
        if not filter_annotations:
            df_relevant = df[
                [*self.feature_names, self._label_column, self._roi_id_colum]
            ]
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
        if set_index:
            df_relevant.set_index(
                [self._roi_id_colum, self._label_column], inplace=True
            )
        return df_relevant

    def save(self):
        """
        Save the classifier and handle overwriting of existing classifier file
        """
        if not self.auto_save:
            # Handle existing classifier file => ask for overwrite
            if not overwrite_check_passed(
                file_path=self._save_destination.value, output_type="classifier"
            ):
                return
        # If the user confirms overwriting the classifier once, keep
        # overwriting it going forward. We want classifier auto-save, just not
        # overwriting of other existing classifiers with the same name.
        self.auto_save = True
        output_path = Path(self._save_destination.value)
        self._classifier.save(output_path)

    def _update_export_destination(self, label_layer: napari.layers.Labels):
        """
        Update the default export destination to the name of the label layer.
        If a base_path was already set, keep it on that base path.

        """
        base_path = Path(self._export_destination.value).parent
        self._export_destination.value = (
            base_path / f"{label_layer.name}_predictions.csv"
        )

    def export_results(self):
        """
        Export classifier results for the current layer if available
        """
        if not overwrite_check_passed(
            file_path=self._export_destination.value, output_type="predictions"
        ):
            return

        predictions = self._last_selected_label_layer.features.loc[
            :, [self._label_column, "prediction", "annotations"]
        ]
        # pylint: disable=C0103
        df = add_annotation_names(
            df=pd.DataFrame(predictions), ClassSelection=self._annotator.ClassSelection
        )

        df.to_csv(self._export_destination.value)
        napari_info(f"Annotations were saved at {self._export_destination.value}")


class LoadClassifierContainer(Container):
    """
    The `LoadClassifierContainer` is a second entry-way to the classifier and
    can launch an appropriate `ClassifierRunContainer`.

    Paramters
    ---------
    viewer: napari.Viewer
        The current napari.Viewer instance

    Attributes
    ----------
    viewer: napari.Viewer
        The current napari.Viewer instance
    _clf_destination: magicgui.widgets.FileEdit
        The file edit widget that allows the user to select a classifier file
    _load_button: magicgui.widgets.PushButton
        The button that launches the `ClassifierRunContainer`
    _filter: magicgui.widgets.RadioButtons
        The radio button widget that allows the user to select the file filter
        to use for selecting the classifier file. See
        https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/issues/36
        for more details.
    _run_container: ClassifierRunContainer
         The `ClassifierRunContainer` that is launched by the `_load_button`
    """

    def __init__(self, viewer: napari.viewer.Viewer):
        self._viewer = viewer
        self._clf_destination = FileEdit(mode="r", filter=None)
        self._filter = RadioButtons(
            value="*.clf",
            choices=["*.clf", "*.pkl", "*"],
            orientation="horizontal",
            label="Filter",
        )
        self._load_button = PushButton(label="Load Classifier")
        self._run_container = None
        super().__init__(
            widgets=[self._clf_destination, self._filter, self._load_button]
        )
        self._load_button.clicked.connect(self.load)
        self._filter.changed.connect(self.set_filter)

    def set_filter(self):
        """
        Updates the filter that is applied to the file edit widget
        """
        self._clf_destination.filter = self._filter.value

    def load(self):
        """
        Load a classifier from a file and start the run container with the
        correct options(already set classifier_save_path and turn on auto_save)
        """
        clf_path = Path(self._clf_destination.value)
        with open(clf_path, "rb") as f:  # pylint: disable=C0103
            clf = pickle.load(f)

        self._run_container = ClassifierRunContainer(
            self._viewer,
            clf,
            classifier_save_path=clf_path,
            auto_save=True,
        )
        self.clear()
        self.append(self._run_container)

        # TODO: Add functionality that loads existing annotations from the
        # classifier and adds them back to the currently open label images


class ClassifierWidget(Container):
    """
    The `ClassifierWidget` is the parent widget and the one that is added as
    a dockwidget. It manages the `ClassifierInitContainer` and the
    `ClassifierRunContainer`.

    Paramters
    ---------
    viewer: napari.Viewer
        The current napari.Viewer instance

    Attributes
    ----------
    viewer: napari.Viewer
        The current napari.Viewer instance
    _run_container: None or ClassifierRunContainer
        The `ClassifierRunContainer` that's started.
    _init_container: None or ClassifierInitContainer
        The `ClassifierInitContainer` that's started.
    """

    def __init__(self, viewer: napari.viewer.Viewer):
        self._viewer = viewer

        self._init_container = None
        self._run_container = None
        self._init_container = None
        self.setup_logging()

        super().__init__(widgets=[])

        self.initialize_init_widget()

    def setup_logging(self):
        # Create a custom handler for napari
        napari_handler = NapariHandler()
        napari_handler.setLevel(logging.INFO)

        # Optionally, set a formatter for the handler
        # formatter = logging.Formatter(
        #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        # )
        # napari_handler.setFormatter(formatter)

        # Get the classifier's logger and add the napari handler to it
        classifier_logger = logging.getLogger("classifier")
        classifier_logger.addHandler(napari_handler)

    def initialize_init_widget(self):
        self._init_container = ClassifierInitContainer(self._viewer)
        self.append(self._init_container)
        self._init_container._initialize_button.clicked.connect(
            self.initialize_run_widget
        )

    def initialize_run_widget(self):
        class_names = self._init_container.get_class_names()
        feature_names = self._init_container.get_selected_features()
        if not feature_names:
            napari_info("No features selected")
            return
        self._run_container = ClassifierRunContainer(
            self._viewer, class_names=class_names, feature_names=feature_names
        )
        self.clear()
        self.append(self._run_container)
