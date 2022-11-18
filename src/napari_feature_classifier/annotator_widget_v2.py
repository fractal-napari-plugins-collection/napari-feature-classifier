from typing import Optional
import numpy as np
import pandas as pd
from magicgui import magic_factory, widgets, magicgui
from napari_plugin_engine import napari_hook_implementation
import napari
from pathlib import Path


# def main():
#     # print(initialize_annotator())
#     viewer = napari.Viewer()
#     # viewer.add_labels(np.zeros((10, 10), dtype=int))
#     # napari_experimental_provide_dock_widget()
#     viewer.add_labels(np.zeros((10, 10), dtype=np.uint16))
#     # viewer.add_label
#     # threshold_widget = threshold()
#     viewer.window.add_dock_widget(initialize_annotator)
#     viewer.show(block=True)
#     # napari_experimental_provide_dock_widget()


# @magicgui(auto_call=True, threshold={'max': 2 ** 16})
# def threshold(
#     data: 'napari.types.ImageData', threshold: int
# ) -> 'napari.types.LabelsData':
#     return (data > threshold).astype(int)
#
# @napari_hook_implementation
# def napari_experimental_provide_dock_widget():
#     return threshold

def _initialize_annotation_stuff(widget):

    @widget.label_layer.changed.connect
    def initialize_label_layer():
        if widget.label_layer.value is not None:
            # TODO: Check if annotator layer already exists and handle that
            # Create an annotator layer
            widget.annotation_layer.value = widget.viewer.value.add_labels(widget.label_layer.value.data, name='annotations')

            # TODO: Check if annotator column already exists
            # Add an annotator column to the label_layer
            unique_labels = np.unique(widget.label_layer.value.data)[1:]
            widget.label_layer.value.features['annotation'] = np.zeros(len(unique_labels))
            widget.label_layer.value.features.index = unique_labels

            print(widget.annotation_layer)
            print(widget.annotation_layer.value)
    #widget.label_layer.value.features['annotation'] = pd.Series(dtype=pd.Int64Dtype)
    # TODO: create layer


# TODO: Make the annotation layer something the user cannot select
@magic_factory(
        classes={'widget_type': 'RadioButtons', 'choices': ['Clear', 'Class 1', 'Class 2']},
        annotation_layer={'visible': False},
        widget_init=_initialize_annotation_stuff,
    )
def initialize_annotator(
        viewer: napari.Viewer,
        label_layer: "napari.layers.Labels",
        annotation_layer: "napari.layers.Labels",
        classes: list[str],
        output_path: Optional[Path] = Path('.') / 'annotation.csv',
):
    pass


    # TODO: All the connection to events
    @label_layer.mouse_drag_callbacks.append
    def toggle_label(_, event):  # pylint: disable-msg=W0613
        """
        Handles user annotations by setting the corresponding classifier
        variables and changing the
        """
        self.annotation_layer.visible = True
        # Need to scale position that event.position returns by the
        # label_layer scale.
        # If scale is (1, 1, 1), nothing changes
        # If scale is anything else, this makes the click still match the
        # correct label
        scaled_position = tuple(
            pos / scale for pos, scale in zip(event.position, self.label_layer.scale)
        )
        label = self.label_layer.get_value(scaled_position)
        if selector.value is None:
            napari_info(
                "No class is selected. Select a class in the classifier widget."
            )
            return
        # Check if background or foreground was clicked. If background was
        # clicked, do nothing (background can't be assigned a class)
        if label == 0 or label is None:
            napari_info("No label clicked.")
            return
        self.annotations[label] = choices.index(selector.value)
        self.update_annotation_colormap(label, choices.index(selector.value))

    # THIS SHOULD NOT BE NECESSARY ANYMORE. GET DIRECTLY FROM widget.classes?
    # @classes.changed.connect
    # def change_choice():
    #     self.annotation_layer.visible = True
    #     self.viewer.layers.selection.clear()
    #     # This doesn't work during testing
    #     try:
    #         self.viewer.layers.selection.add(self.label_layer)
    #     except ValueError:
    #         pass

    # @export_button.changed.connect
    # def export_annotations():
    #     if not str(export_path.value).endswith(".csv"):
    #         warnings.warn(
    #             "The export path does not lead to a .csv file. This "
    #             "export function will export in .csv format anyway"
    #         )

    #     # Check if file already exists
    #     if os.path.exists(Path(export_path.value)):
    #         msg_box = QMessageBox()
    #         msg_box.setText(
    #             f"A csv export with the name {Path(export_path.value).name}"
    #             " already exists. This will overwrite it."
    #         )
    #         msg_box.setWindowTitle("Overwrite Export?")
    #         msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    #         answer = msg_box.exec()
    #         if answer == QMessageBox.Cancel:
    #             return
    #     napari_info("Exporting classifier results")

    #     self.annotations.to_csv(export_path.value)

    # @label_layer.bind_key("b", overwrite=True)
    # def toggle_selection_layer_visibility(layer):  # pylint: disable-msg=W0613
    #     self.annotation_layer.visible = not self.annotation_layer.visible

    # @self.label_layer.bind_key("v", overwrite=True)
    # def toggle_label_layer_visibility(layer):
    #     # Only set opacity to 0. Otherwise layer is not clickable anymore.
    #     if self.label_layer.opacity > 0:
    #         self.label_layer.opacity = 0.0
    #     else:
    #         self.label_layer.opacity = 0.8

    # def set_class_n(event, n):
    #     selector.value = choices[n]
    #     change_choice()

    # # keybindings for the available classes (0 = deselect)
    # for i in range(self.n_classes + 1):
    #     set_class = partial(set_class_n, n=i)
    #     self.label_layer.bind_key(str(i), set_class)

    # return container




# if __name__ == '__main__':
#     main()
