"""Classifier container widget for napari"""

import logging
import pickle
from pathlib import Path

import napari
import napari.layers
import napari.viewer
import pandas as pd
from magicgui.widgets import (
    Container,
    FileEdit,
    Label,
    PushButton,
    RadioButtons,
    Select,
)

from napari_feature_classifier.annotator_init_widget import LabelAnnotatorTextSelector
from napari_feature_classifier.annotator_widget import (
    CollapsibleSection,
    LabelAnnotator,
    get_class_selection,
)
from napari_feature_classifier.classifier import Classifier
from napari_feature_classifier.classifier_runner import (
    ClassifierRunner,
    PredictionLayerManager,
)
from napari_feature_classifier.utils import (
    NapariHandler,
    add_annotation_names,
    get_selected_or_valid_label_layer,
    get_valid_label_layers,
    napari_info,
    overwrite_check_passed,
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
        layer_name = (
            str(self._last_selected_label_layer)
            if self._last_selected_label_layer
            else "None"
        )
        self.last_selected_layer_label = Label(
            value=f"Select features from: {layer_name}"
        )
        self._feature_combobox = Select(
            choices=self.get_feature_options(self._last_selected_label_layer),
            allow_multiple=True,
            label="",
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
            ],
            labels=False,
        )
        self.native.layout().setContentsMargins(0, 0, 0, 0)
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
            self.last_selected_layer_label.value = (
                f"Select features from: {self._last_selected_label_layer}"
            )
            self._feature_combobox.choices = self.get_feature_options(
                self._last_selected_label_layer
            )
            # pylint: disable=W0212
            self._feature_combobox._default_choices = self.get_feature_options(
                self._last_selected_label_layer
            )


class ClassifierExportPanel(Container):
    """
    Owns the save and export UI for a trained classifier.

    Parameters
    ----------
    classifier: Classifier
        The classifier to save.
    annotator: LabelAnnotator
        Used to resolve class names when exporting predictions.
    initial_layer: napari.layers.Labels
        The initially selected label layer (used for default file names).
    classifier_save_path: Optional[str]
        Pre-filled save path for the classifier file.
    auto_save: bool
        If True, skip overwrite confirmation on save.
    label_column: str
        Column name for object labels in layer.features.
    """

    def __init__(
        self,
        classifier: Classifier,
        annotator: "LabelAnnotator",
        initial_layer: napari.layers.Labels,
        classifier_save_path: str | None = None,
        auto_save: bool = False,
        label_column: str = "label",
    ):
        self._classifier = classifier
        self._annotator = annotator
        self._last_selected_label_layer = initial_layer
        self._label_column = label_column
        self.auto_save = auto_save

        self._save_path = Path(
            classifier_save_path or f"{initial_layer}_classifier.clf"
        )
        self._export_path = Path(f"{initial_layer}_predictions.csv")

        self._save_button = PushButton(text="Save Classifier As…")
        self._export_button = PushButton(text="Export Results As…")

        self._saving_section = CollapsibleSection(
            "Saving & Export",
            [self._save_button, self._export_button],
            collapsed=True,
        )

        super().__init__(widgets=[self._saving_section], labels=False)
        self.native.layout().setContentsMargins(0, 0, 0, 0)
        self._save_button.clicked.connect(self._on_save_as_clicked)
        self._export_button.clicked.connect(self._on_export_clicked)

    def update_selected_layer(self, label_layer: napari.layers.Labels) -> None:
        """Update the tracked layer and refresh the export path."""
        self._last_selected_label_layer = label_layer
        self._update_export_destination(label_layer)

    def save(self) -> None:
        """Save the classifier to _save_path, with overwrite check on first save."""
        if not self.auto_save:
            if not overwrite_check_passed(
                file_path=self._save_path, output_type="classifier"
            ):
                return
        self.auto_save = True
        self._classifier.save(self._save_path)
        napari_info(f"Classifier saved at {self._save_path}")

    def export_results(self) -> None:
        """Export classifier predictions for the currently selected layer."""
        predictions = self._last_selected_label_layer.features.loc[
            :, [self._label_column, "prediction", "annotations"]
        ]
        # pylint: disable=C0103
        df = add_annotation_names(
            df=pd.DataFrame(predictions),
            ClassSelection=self._annotator.ClassSelection,
        )
        df.to_csv(self._export_path)
        napari_info(f"Annotations were saved at {self._export_path}")

    def _on_save_as_clicked(self) -> None:
        """Open a Save As dialog and save the classifier to the chosen path."""
        from qtpy.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(  # type: ignore[misc]
            None,
            "Save Classifier",
            str(self._save_path),
            "Classifier files (*.clf);;All files (*)",
        )
        if not path:
            return
        self._save_path = Path(path)
        self.auto_save = True
        self._classifier.save(self._save_path)
        napari_info(f"Classifier saved at {self._save_path}")

    def _on_export_clicked(self) -> None:
        """Open a Save As dialog and export predictions to the chosen path."""
        from qtpy.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(  # type: ignore[misc]
            None,
            "Export Results",
            str(self._export_path),
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        self._export_path = Path(path)
        self.export_results()

    def _update_export_destination(self, label_layer: napari.layers.Labels) -> None:
        """Update the default export path to match the selected layer name."""
        base_path = self._export_path.parent
        self._export_path = base_path / f"{label_layer.name}_predictions.csv"


class ClassifierRunContainer(Container):
    """
    Coordinator widget that wires together the annotator, classifier
    runner, prediction layer manager, and export panel.

    The `ClassifierRunContainer` can be initialized with either an existing
    classifier or with class_names + feature_names.

    Parameters
    ----------
    viewer: napari.Viewer
        The current napari.Viewer instance.
    classifier: Optional[Classifier]
        An existing classifier to resume from. If omitted, class_names and
        feature_names must be provided.
    class_names: Optional[list[str]]
        Class names for a new classifier (ignored when classifier is given).
    feature_names: Optional[list[str]]
        Feature names for a new classifier (ignored when classifier is given).
    classifier_save_path: Optional[str]
        Pre-filled save path passed to ClassifierExportPanel.
    auto_save: Optional[bool]
        If True, skip overwrite confirmation on first save.

    Attributes
    ----------
    _viewer: napari.Viewer
    _classifier: Classifier
    _runner: ClassifierRunner
    _prediction_manager: PredictionLayerManager
    _annotator: LabelAnnotator
    _export_panel: ClassifierExportPanel
    _run_button: magicgui.widgets.PushButton
    """

    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        classifier: Classifier | None = None,
        class_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        classifier_save_path: str | None = None,
        auto_save: bool | None = False,
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

        self._runner = ClassifierRunner(self._viewer, self._classifier)
        self._prediction_manager = PredictionLayerManager(self._viewer)

        self._annotator = LabelAnnotator(
            self._viewer,
            get_class_selection(class_names=self.class_names),
            annotation_callbacks=[self._update_counts],
            class_colors=self._classifier._class_colors,
            on_names_changed=self._on_class_names_changed,
            on_colors_changed=self._on_class_colors_changed,
        )

        # Wire per-class colors into the prediction layer renderer
        self._prediction_manager._color_resolver = (
            self._annotator._class_selector.get_color
        )

        self._export_panel = ClassifierExportPanel(
            classifier=self._classifier,
            annotator=self._annotator,
            initial_layer=self._last_selected_label_layer,
            classifier_save_path=classifier_save_path,
            auto_save=auto_save or False,
        )

        feature_names = self._classifier.get_feature_names()
        self._feature_list = Select(
            choices=feature_names,
            value=feature_names,
            allow_multiple=True,
            label="",
        )
        # Make read-only at the Qt level: no selection, no focus, no interaction.
        # Do NOT use .enabled = False — that propagates through magicgui's container
        # chain and disables the whole plugin.
        from qtpy.QtCore import Qt  # type: ignore[attr-defined]
        from qtpy.QtWidgets import QAbstractItemView

        self._feature_list.native.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self._feature_list.native.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._feature_section = CollapsibleSection(
            "Features", [self._feature_list], collapsed=True
        )

        self._run_button = PushButton(text="Run Classifier")

        super().__init__(
            widgets=[
                self._annotator,
                self._feature_section,
                self._run_button,
                self._export_panel,
            ],
            labels=False,
        )
        self.native.layout().setContentsMargins(0, 0, 0, 0)
        self._prediction_manager.setup(self._last_selected_label_layer)
        # Restore any stored annotations for the initially selected layer
        self._restore_annotations(self._last_selected_label_layer)
        # Set the label selection to a valid label layer => Running into proxy bug
        self._viewer.layers.selection.active = self._last_selected_label_layer
        self._run_button.clicked.connect(self.run)
        self._viewer.layers.selection.events.changed.connect(self.selection_changed)
        # Initialise counts now that the panel is fully wired
        self._update_counts()

    def run(self):
        """
        Run the classifier pipeline in a background thread to keep the UI
        responsive during feature collection and training.

        Feature collection and training run off the main thread.  All napari
        layer writes (predictions, colormap, save) are dispatched back to the
        main thread via the worker's `returned` / `errored` signals.
        """
        from napari.qt.threading import thread_worker

        self._run_button.enabled = False
        self._run_button.text = "Running…"

        @thread_worker(ignore_errors=True)
        def _train():
            self._runner.add_features_to_classifier()
            f1 = self._classifier.train()  # raises ValueError on bad input
            return f1

        self._run_worker = _train()
        # Connect to self methods (QObject) so PyQt uses QueuedConnection for
        # cross-thread dispatch, ensuring callbacks run in the main thread.
        self._run_worker.returned.connect(self._on_run_done)
        self._run_worker.errored.connect(self._on_run_error)
        self._run_worker.start()

    def _on_run_done(self, _f1) -> None:
        """Called in the main thread when background training succeeds."""
        assert isinstance(self._last_selected_label_layer, napari.layers.Labels)
        self._runner.make_predictions()
        self._prediction_manager.setup(self._last_selected_label_layer)
        self._prediction_manager.set_visible(True)
        self._export_panel.save()
        self._update_counts()
        self._run_button.text = "Run Classifier"
        self._run_button.enabled = True

    def _on_run_error(self, exc: Exception) -> None:
        """Called in the main thread when background training raises."""
        if isinstance(exc, ValueError):
            napari_info(
                "Training failed. A typical reason are not having "
                "enough annotations. \nThe error message was: "
                f"{exc}"
            )
        else:
            napari_info(f"Unexpected error during training: {exc}")
        self._run_button.text = "Run Classifier"
        self._run_button.enabled = True

    def selection_changed(self):
        """
        Check if the selection change results in a valid label layer being
        selected. If so, sync prediction layer and export panel to it.
        """
        active = self._viewer.layers.selection.active
        if (
            active
            and active in get_valid_label_layers(viewer=self._viewer)
            and isinstance(active, napari.layers.Labels)
        ):
            self._last_selected_label_layer = active
            # LabelAnnotator.selection_changed fires first (connected earlier),
            # so the annotations column already exists by the time we get here.
            self._restore_annotations(active)
            self._prediction_manager.sync(active)
            self._export_panel.update_selected_layer(active)
            self._update_counts()

    def _restore_annotations(self, layer: napari.layers.Labels) -> None:
        """
        Restore stored annotations from classifier._data into the layer and
        re-render the annotation colormap if any were written back.
        """
        restored = self._runner.restore_annotations_to_layer(layer)
        if restored:
            # Re-run _init_annotation to refresh the colormap; it is idempotent
            # when the annotations column already exists.
            self._annotator._init_annotation(layer)
            self._update_counts()

    def _on_class_names_changed(self, new_names: list[str]) -> None:
        """Sync classifier class names when the user renames a class."""
        self._classifier._class_names = list(new_names)

    def _on_class_colors_changed(
        self, class_index: int, rgba: tuple[float, float, float, float]
    ) -> None:
        """Persist color edits to the classifier and refresh the prediction layer."""
        self._classifier._class_colors[class_index] = rgba
        # Re-render the prediction layer with the updated color
        try:
            self._prediction_manager.sync(self._last_selected_label_layer)
        except RuntimeError:
            pass  # prediction layer not yet set up

    def _update_counts(self) -> None:
        self._annotator._class_selector.update_counts(self._get_annotation_counts())

    def _get_annotation_counts(self) -> dict[str, int]:
        """
        Return per-class annotation counts merging live open layers with
        historical data from classifier._data (closed images included).
        """
        # Collect live annotations from currently open label layers
        open_roi_ids: dict[str, pd.Series] = {}
        for layer in get_valid_label_layers(self._viewer):
            if "annotations" not in layer.features.columns:
                continue
            if "roi_id" in layer.features.columns:
                unique = layer.features["roi_id"].unique()
                if len(unique) != 1:
                    continue
                roi_id = unique[0]
            else:
                roi_id = layer.name
            open_roi_ids[roi_id] = layer.features["annotations"]  # type: ignore[assignment]

        # Add historical annotations for roi_ids no longer open
        parts = list(open_roi_ids.values())
        if len(self._classifier._data) > 0:
            clf_ann = self._classifier._data["annotations"]
            clf_rois = self._classifier._data.index.get_level_values("roi_id")
            closed = clf_ann[~clf_rois.isin(open_roi_ids)]
            if len(closed):
                parts.append(closed)  # type: ignore[arg-type]

        if not parts:
            return {name: 0 for name in self._classifier.get_class_names()}

        counts = pd.concat(parts, ignore_index=True).dropna().value_counts()
        return {
            name: int(counts.get(float(i + 1), 0))  # type: ignore[arg-type]
            for i, name in enumerate(self._classifier.get_class_names())
        }


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
        from magicgui.types import FileDialogMode

        self._clf_destination = FileEdit(mode=FileDialogMode.EXISTING_FILE, filter=None)
        self._filter = RadioButtons(
            value="*.clf",  # pyright: ignore[reportCallIssue]
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
        clf_path = Path(self._clf_destination.value)  # type: ignore[arg-type]
        with open(clf_path, "rb") as f:  # pylint: disable=C0103
            clf = pickle.load(f)

        try:
            self._run_container = ClassifierRunContainer(
                self._viewer,
                clf,
                classifier_save_path=str(clf_path),
                auto_save=True,
            )
        except NotImplementedError:
            napari_info(
                "Create a label layer with a feature dataframe before loading "
                "the classifier"
            )
            return
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
        assert self._init_container is not None
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
