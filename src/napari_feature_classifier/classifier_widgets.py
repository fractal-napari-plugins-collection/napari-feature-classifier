"""Classfier widget code"""
import warnings
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from magicgui import magic_factory
from napari import Viewer
from magicgui import widgets
from matplotlib.colors import ListedColormap
from napari.utils.notifications import show_info
from .utils import get_df
from .classifier import Classifier


def _init_classifier(widget):
    """
    Classifier Initialization Widget initialization

    Parameters
    ----------
    widget: napari widget
    """
    def get_feature_choices(*args):
        """
        Function loading the column names of the widget dataframe
        """
        try:
            df = get_df(widget.feature_path.value)
            return list(df.columns)
        except IOError:
            return [""]

    # set feature and label_column "default choices"
    # to be a function that gets the column names of the
    # currently loaded dataframe
    widget.feature_selection._default_choices = get_feature_choices
    widget.label_column._default_choices = get_feature_choices

    @widget.feature_path.changed.connect
    def update_df_columns():
        """
        Handles updating of dropdown options and setting defaults
        """
        # ...reset_choices() calls the "get_feature_choices" function above
        # to keep them updated with the current dataframe
        widget.feature_selection.reset_choices()
        widget.label_column.reset_choices()
        features = widget.label_column.choices
        if "label" in features:
            widget.label_column.value = "label"
        elif "Label" in features:
            widget.label_column.value = "Label"
        elif "index" in features:
            widget.label_column.value = "index"

        # If a user provides a custom property called "feature_selection" for
        # the label layer and the content of this property is a valid choice,
        # set the feature selection to that property
        if "feature_selection" in widget.label_layer.value.properties:
            if (
                widget.label_layer.value.properties["feature_selection"]
                in widget.feature_selection.choices
            ):
                widget.feature_selection.value = widget.label_layer.value.properties[
                    "feature_selection"
                ]

    @widget.label_layer.changed.connect
    def update_paths():
        """
        Handles changing label_layer inputs

        If a user provides a custom property called "feature_path" for
        the label layer, set the feature_path to that property
        """
        if "feature_path" in widget.label_layer.value.properties:
            widget.feature_path.value = widget.label_layer.value.properties["feature_path"]


@magic_factory(
    call_button="Initialize Classifier",
    label_layer={"label": "Label Layer:"},
    feature_path={"label": "Feature Path:"},
    classifier_name={"label": "Classifier Name:"},
    feature_selection={
        "choices": [""],
        "allow_multiple": True,
        "label": "Feature Selection:"
    },
    label_column={"choices": [""], "label": "Label Column:"},
    widget_init=_init_classifier,
)
def initialize_classifier(
    viewer: Viewer,
    label_layer: "napari.layers.Labels",
    feature_path: Path,
    classifier_name="test",
    feature_selection=[""],
    label_column="",
):
    """
    Launches classifier initialization dockwidget

    Parameters
    ----------
    viewer: napari.Viewer
        The current napari.Viewer instance
    label_layer: napari.layers.Labels
        The napari label layer on which objects shall be classified
    feature_path: pathlib.Path
        Path to the .csv file that contains the measurements used for
        quantification and a column of label integers
    classifier_name: str
        Name as which the classifier will be saved. If the default test is
        chosen, the classifier will be saved as test.clf in the current working
        directory
    feature_selection: list
        List of features that can be selected to classify the objects
    label_column: str
        Column name of the column in the feature_path csv file containing the
        label values
    """
    # TODO: Check whether features are associated with the Labels layer in the
    # new napari convention (as a dataframe). Use them if they are,
    # otherwise load csv
    if not str(feature_path).endswith(".csv"):
        warnings.warn(
            "The feature_path does not lead to a .csv file. This "
            "classifier requires the data to be save in a .csv "
            "file that is readable with pd.read_csv()"
        )

    site_df = get_df(feature_path)
    site_df["path"] = feature_path
    index_columns = ("path", label_column)
    site_df = site_df.set_index(list(index_columns))

    if os.path.exists(classifier_name + ".clf"):
        # TODO: Add a warning if a classifier with this name already exists =>
        # shall it be overwritten? => Confirmation box
        warnings.warn(
            "A classifier with this name already exists and will be overwritten"
        )
    clf = Classifier(
        name=classifier_name,
        features=site_df,
        training_features=feature_selection,
        index_columns=index_columns,
    )

    # TODO: Check whether features were selected. Pop up a warning if no
    # features were selected
    if len(feature_selection) < 1:
        warnings.warn(
            "No features were selected for the classifier. Please select '\
            'features before initializing the classifier"
        )
    else:
        ClassifierWidget(clf, label_layer, feature_path, viewer)


def _init_load_classifier(widget):
    # TODO: Add an option to check the current working directory for .clf files?
    #       As an option if no classifier_path is provided as a property
    # Inputs always update with properties when label layer is changed.
    @widget.label_layer.changed.connect
    def update_paths():
        if "classifier_path" in widget.label_layer.value.properties:
            widget.classifier_path.value = widget.label_layer.value.properties[
                "classifier_path"
            ]
        if "feature_path" in widget.label_layer.value.properties:
            widget.feature_path.value = widget.label_layer.value.properties["feature_path"]


@magic_factory(
    call_button="Load Classifier",
    label_layer={"label": "Label Layer:"},
    classifier_path={"label": "Classifier Name:"},
    feature_path={"label": "Feature Path:"},
    widget_init=_init_load_classifier
)
def load_classifier(
    viewer: Viewer,
    label_layer: "napari.layers.Labels",
    classifier_path: Path,
    feature_path: Path,
):
    """
    Launches classifier loading dockwidget

    Loads an existing classifier from a .clf file with the set options for
    feature_selection

    Parameters
    ----------
    viewer: napari.Viewer
        The current napari.Viewer instance
    label_layer: napari.layers.Labels
        The napari label layer on which objects shall be classified
    classifier_path: pathlib.Path
        Path to an existing classifier .clf file
    feature_path: pathlib.Path
        Path to the .csv file that contains the measurements used for
        quantification and a column of label integers
    """
    # TODO: Add option to add new features to the classifier that were not
    # added at initialization => unsure where to do this. Should it also be
    # possible when initializing a classifier?
    # TODO: Add ability to see currently selected features (-> part of being
    # able to change the features)
    if not str(feature_path).endswith(".csv"):
        warnings.warn(
            "The feature_path does not lead to a .csv file. This "
            "classifier requires the data to be save in a .csv "
            "file that is readable with pd.read_csv()"
        )

    if not str(classifier_path).endswith(".clf"):
        warnings.warn(
            "The classifier_path does not lead to a .clf file. This "
            "plugin only works with classifiers created by its own "
            "classifier class that are saved as .clf files"
        )

    with open(classifier_path, "rb") as f:
        clf = pickle.loads(f.read())

    training_features = clf.training_features
    site_df = get_df(feature_path)
    site_df["path"] = feature_path
    index_columns = clf.index_columns
    # Catches if new data frame doesn't contain the index columns
    assert all(
        [index_column in site_df.columns for index_column in index_columns]
    ), "These two columns are not available in the current dataframe: {}".format(
        index_columns
    )
    site_df = site_df.set_index(list(index_columns))

    clf.add_data(
        site_df, training_features=training_features, index_columns=index_columns
    )

    ClassifierWidget(clf, label_layer, feature_path, viewer)


class ClassifierWidget:
    def __init__(self, clf, label_layer, feature_path, viewer):
        self.clf = clf
        self.clf.save()
        self.label_layer = label_layer
        self.feature_path = feature_path
        self.viewer = viewer

        # Parameters for the colormaps
        # TODO: Generalize number of classes & colormap
        self.nb_classes = 4
        self.cmap = ListedColormap(
            [
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0, 1.0),
                (0.0, 0.0, 1.0, 1.0),
                (1.0, 0.0, 1.0, 1.0),
            ]
        )

        # Create a selection & prediction layer
        # TODO: Handle state when those layers were already created. Replace
        # them otherwise?
        # https://napari.org/guides/stable/magicgui.html#updating-an-existing-layer
        if "prediction" in viewer.layers:
            viewer.layers.remove("prediction")
        if "selection" in viewer.layers:
            viewer.layers.remove("selection")
        self.prediction_layer = viewer.add_labels(
            label_layer.data, name="prediction", opacity=1.0, scale=label_layer.scale
        )
        self.selection_layer = viewer.add_labels(
            label_layer.data, name="selection", opacity=1.0, scale=label_layer.scale
        )
        self.colordict = self.create_label_colormap(
            self.selection_layer, clf.train_data, "train"
        )
        self.create_label_colormap(self.prediction_layer, clf.predict_data, "predict")
        self.viewer.layers.selection.clear()
        self.viewer.layers.selection.add(label_layer)

        widget = self.create_selector_widget(self.label_layer)

        # TODO: Find a new way to do remove widget of another existing
        # classifier. Currently triggers a deprecation
        # warning for napari 0.5
        # If a widget already exists for the classifier with the same name,
        # remove it
        # TODO: Is there a way to get rid of other class selection windows?
        # I don't have a pointer to them and they could have arbitrary names
        # try:
        #     if self.clf.name in viewer.window._dock_widgets:
        #         viewer.window.remove_dock_widget(viewer.window._dock_widgets[self.clf.name])
        # except:
        #     # If the API for getting dock_widgets changes, just ignore this.
        #     # This is optional functionality
        #     pass

        # add widget to napari
        viewer.window.add_dock_widget(widget, area="right", name=clf.name)

    def create_selector_widget(self, label_layer):
        # TODO: Generalize this. Instead of 0, 1, 2, 3, 4: Arbitrary class
        # numbers. Ability to add classes & name them?
        choices = ["Deselect", "Class 1", "Class 2", "Class 3", "Class 4"]
        selector = widgets.RadioButtons(
            choices=choices, label="Selection Class:", value="Class 1"
        )
        save_button = widgets.PushButton(value=True, text="Save Classifier")
        run_button = widgets.PushButton(value=True, text="Run Classifier")
        export_path = widgets.LineEdit(
            value=Path(os.getcwd()) / "Classifier_output.csv", label="Export Name:"
        )
        export_button = widgets.PushButton(value=True, text="Export Classifier Result")
        container = widgets.Container(
            widgets=[selector, run_button, save_button, export_path, export_button]
        )

        @label_layer.mouse_drag_callbacks.append
        def toggle_label(obj, event):
            # TODO: Add a warning when user clicks while the wrong layer is
            # selected?
            self.selection_layer.visible = True
            # Need to scale position that event.position returns by the
            # label_layer scale.
            # If scale is (1, 1, 1), nothing changes
            # If scale is anything else, this makes the click still match the
            # correct label
            scaled_position = tuple(
                pos / scale for pos, scale in zip(event.position, label_layer.scale)
            )
            label = label_layer.get_value(scaled_position)
            if selector.value is None:
                show_info(
                    "No class is selected. Select a class in the classifier widget."
                )
            # Check if background or foreground was clicked. If background was
            # clicked, do nothing (background can't be assigned a class)
            elif label == 0 or label is None:
                pass
            else:
                # Check if the label exists in the current dataframe.
                # Otherwise, do nothing
                if (self.feature_path, label) in self.clf.train_data.index:
                    # Assign name of class
                    # self.clf.train_data.loc[(feature_path, label)] = selector.value
                    # Assign a numeric value to make it easier
                    # (colormap currently only supports this mode)
                    self.clf.train_data.loc[(self.feature_path, label)] = choices.index(
                        selector.value
                    )
                    self.update_label_colormap(
                        self.selection_layer, label, choices.index(selector.value)
                    )
                    self.clf.is_trained = False
                else:
                    show_info(
                        "The data that was provided to the classifier "
                        "does not contain an object with index {}. "
                        "Thus, this object cannot be included in the "
                        "classifier".format(label)
                    )

        @selector.changed.connect
        def change_choice():
            self.selection_layer.visible = True
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.add(self.label_layer)

        @label_layer.bind_key("s", overwrite=True)
        @save_button.changed.connect
        def save_classifier():
            show_info("Saving classifier")
            self.clf.save()

        @export_button.changed.connect
        def export_classifier():
            # TODO: Check if file path ends in csv.
            # If not, give a warning dialogue with the option to cancel or add a .csv
            if not str(export_path.value).endswith(".csv"):
                warnings.warn(
                    "The export path does not lead to a .csv file. This "
                    "export function will export in .csv format anyway"
                )

            # TODO: Check if file already exists
            # If it does, warning dialog with option to overwrite (default) or cancel

            show_info("Exporting classifier results")
            self.clf.export_results_single_site(export_path.value)

        @label_layer.bind_key("t", overwrite=True)
        @run_button.changed.connect
        def run_classifier(key: str):
            # Check if the classifer contains any training data
            if len(self.clf.train_data['train'].unique()) > 1:
                # TODO: Add Run mode? Fuzzy (i.e. trained on everything),
                # Cross-validated, train/test split
                show_info("Running classifier")
                self.clf.train()
                self.create_label_colormap(
                    self.prediction_layer, self.clf.predict_data, "predict"
                )
                self.clf.save()
                self.selection_layer.visible = False
                self.prediction_layer.visible = True
            else:
                warnings.warn("You need to include some annotations to run "
                              "the classifier")

        @label_layer.bind_key("o", overwrite=True)
        def toggle_selection(layer):
            current = self.selection_layer.visible
            self.selection_layer.visible = not current

        @label_layer.bind_key("p", overwrite=True)
        def toggle_selection(layer):
            current = self.prediction_layer.visible
            self.prediction_layer.visible = not current

        @label_layer.bind_key("v", overwrite=True)
        def toggle_selection():
            # Toggling off the label layer would be inconvenient
            # (can't click on it anymore)
            # => just toggle the opacity to 0
            opacity = label_layer.opacity
            if opacity > 0:
                label_layer.opacity = 0.0
            else:
                label_layer.opacity = 0.8

        @label_layer.bind_key("0", overwrite=True)
        def set_class_0(event):
            selector.value = choices[0]
            change_choice()

        @label_layer.bind_key("1", overwrite=True)
        def set_class_1(event):
            selector.value = choices[1]
            change_choice()

        @label_layer.bind_key("2", overwrite=True)
        def set_class_2(event):
            selector.value = choices[2]
            change_choice()

        @label_layer.bind_key("3", overwrite=True)
        def set_class_3(event):
            selector.value = choices[3]
            change_choice()

        @label_layer.bind_key("4", overwrite=True)
        def set_class_4(event):
            selector.value = choices[4]
            change_choice()

        return container

    def update_label_colormap(self, curr_label_layer, label, new_class):
        # This is still kinda laggy on large dataset.
        # Is there a way to not send a whole new colormap, but just change
        # the colormap in one place?
        # See here for discussion on this topic:
        # https://forum.image.sc/t/napari-layer-colormaps-update-individual-objects-only/52547
        # And here for the napari issue:
        # https://github.com/napari/napari/issues/2380
        self.colordict[label] = self.cmap(new_class / self.nb_classes)
        curr_label_layer.color = self.colordict
        # Directly change just the color of the one object, not replacing the
        # whole colormap
        # curr_label_layer.color[label] = self.cmap(new_class/self.nb_classes)
        # Doesn't do anything. Color doesn't update.

    def create_label_colormap(self, curr_label_layer, df, feature):
        site_df = df[df.index.isin([self.feature_path], level=0)]
        site_df.index = site_df.index.droplevel()
        colors = self.cmap(site_df[feature] / self.nb_classes)
        colordict = dict(zip(site_df.index, colors))
        curr_label_layer.color = colordict
        return colordict
