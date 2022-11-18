"""Classfier widget code"""
import warnings
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

import napari
from napari import Viewer  # pylint: disable-msg=E0611
from magicgui import magic_factory, widgets
from matplotlib.colors import ListedColormap
from qtpy.QtWidgets import QMessageBox  # pylint: disable-msg=E0611

from src.napari_feature_classifier.annotator_widget import AnnotatorWidget
from src.napari_feature_classifier.classifier import Classifier, napari_info
from src.napari_feature_classifier.data import ClassifierData
import h5py


def main():
    fn = r"Z:\hmax\Zebrafish\20211119_cyclerDATA_compressed\20211119_ABpanelTestset_3pair_3.5h_1_s6.h5"
    with h5py.File(fn) as f:
        sytox = f['ch_00/1'][:, 512:1024, 512:1024]
        lbl = f['lbl_nuc'][:, 511:1024, 512:1024]
    viewer = napari.Viewer()
    viewer.add_image(sytox, scale=(1.0, 0.65, 0.65))
    lbl_layer = viewer.add_labels(lbl, scale=(1.0, 0.65, 0.65))

    feature_path = r"C:\Users\hessm\Documents\Programming\Python\classifier_demo\featuresNucs_3pair_3.5h_1_s6.csv"
    training_feature_names = ['Roundness', 'NumberOfPixels']

    widget = ClassifierWidget(lbl_layer, feature_path, 'mito', training_feature_names, 'Label', viewer)

    viewer.show(block=True)


class ClassifierInitializer:
    pass


class ClassifierLoader:
    pass


class ClassifierWidget:
    def __init__(self, label_layer, feature_path, classifier_name, training_feature_names, label_column, viewer):
        data = ClassifierData.from_path(feature_path, index_columns=['filename_prefix', label_column],
                                        training_feature_names=training_feature_names)
        self.clf = Classifier(name=classifier_name, data=data, training_features=training_feature_names)
        self.annotator_widget = AnnotatorWidget(label_layer, viewer, n_classes=3)
        self.annotation_layer = self.annotator_widget.annotation_layer
        self.label_layer = label_layer

        # Create annotation layer (overwriting existing ones).
        if "prediction" in viewer.layers:
            viewer.layers.remove("prediction")
        self.prediction_layer = viewer.add_labels(
            label_layer.data, name="prediction", opacity=1.0, scale=label_layer.scale, translate=label_layer.translate,
        )
        self.prediction_layer.color[None] = np.array([0, 0, 0, 0.001], np.float32)
        self.prediction_layer.color_mode = 'direct'
        self.prediction_layer.editable = False

        widget = self.create_classifier_widget()
        viewer.window.add_dock_widget(widget, area="right", name="Classifier")

    def create_classifier_widget(self):
        save_path = widgets.FileEdit(
            value=Path(os.getcwd()) / (self.clf.name + ".clf"),
            label="Save Classifier As:",
            mode="w",
        )
        save_button = widgets.PushButton(value=True, text="Save Classifier")
        run_button = widgets.PushButton(value=True, text="Run Classifier")
        export_path = widgets.FileEdit(
            value=Path(os.getcwd()) / "Classifier_output.csv",
            label="Export Name:",
            mode="w",
            filter="*.csv",
        )
        export_button = widgets.PushButton(value=True, text="Export Classifier Result")
        container = widgets.Container(
            widgets=[
                run_button,
                save_path,
                save_button,
                export_path,
                export_button,
            ]
        )

        @self.label_layer.bind_key("s", overwrite=True)
        @save_button.changed.connect
        def save_classifier():
            # Handle name changes
            classifier_name = Path(save_path.value).name
            directory = Path(save_path.value).parent
            napari_info("Saving classifier")
            self.clf.save(new_name=classifier_name, directory=directory)

        @export_button.changed.connect
        def export_classifier():
            # TODO: Check if file path ends in csv.
            # If not, give a warning dialogue with the option to cancel or add a .csv
            if not str(export_path.value).endswith(".csv"):
                warnings.warn(
                    "The export path does not lead to a .csv file. This "
                    "export function will export in .csv format anyway"
                )

            # Check if file already exists
            if os.path.exists(Path(export_path.value)):
                msg_box = QMessageBox()
                msg_box.setText(
                    f"A csv export with the name {Path(export_path.value).name}"
                    " already exists. This will overwrite it."
                )
                msg_box.setWindowTitle("Overwrite Export?")
                msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                answer = msg_box.exec()
                if answer == QMessageBox.Cancel:
                    return
            napari_info("Exporting classifier results")
            self.clf.export_results(export_path.value)

        @self.label_layer.bind_key("t", overwrite=True)
        @run_button.changed.connect
        def run_classifier(key: str):  # pylint: disable-msg=W0613
            self.clf.data.update_annotations(self.annotator_widget.annotations,
                                             site_id='20211119_ABpanelTestset_3pair_3.5h_1_s6')
            self.clf.train()
            self.clf.save()
            self.update_prediction_colormap(self.clf.data.predictions)

            # # Check if the classifer contains any training data
            # if len(self.clf.train_data["train"].unique()) > 1:
            #     # TODO: Add Run mode? Fuzzy (i.e. trained on everything),
            #     # Cross-validated, train/test split
            #     self.clf.train()
            #     self.create_label_colormap(
            #         self.prediction_layer, self.clf.predict_data, "predict"
            #     )
            #     self.clf.save()
            #     self.annotation_layer.visible = False
            #     self.prediction_layer.visible = True
            # else:
            #     warnings.warn(
            #         "You need to include some annotations to run " "the classifier"
            #     )

        @self.label_layer.bind_key("p", overwrite=True)
        def toggle_selection_pred_layer(layer):  # pylint: disable-msg=W0613
            current = self.prediction_layer.visible
            self.prediction_layer.visible = not current

        return container

    def update_prediction_colormap(self, predictions: pd.Series):
        for label, prediction in predictions.items():
            self.prediction_layer.color[label[1]] = self.annotator_widget.cmap(prediction)
        self.prediction_layer.color_mode = 'direct'


if __name__ == '__main__':
    main()
