from pathlib import Path

import imageio
import napari
import napari.layers
import napari.viewer
import numpy as np
import pandas as pd
from magicgui.widgets import Container, Label
from napari.utils.notifications import show_info

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

    # Add the widget directly via code:
    # label_selector_widget = LabelSelector(viewer)
    # viewer.window.add_dock_widget(label_selector_widget)

    viewer.show(block=True)


class LabelSelector(Container):
    def __init__(
        self,
        viewer: napari.viewer.Viewer,
    ):
        self._viewer = viewer
        self.label = Label(label='Test')
        super().__init__(
            widgets=[
                self.label
            ]
        )
        self._last_selected_label_layer = self._viewer.layers[1]
        annotation_layer = self._viewer.add_labels(
            self._last_selected_label_layer.data,
            scale=self._last_selected_label_layer.scale,
            name="Annotations",
        )
        self._viewer.layers.selection.active = self._viewer.layers[0]
        print(f'Selected Layer at the end: {self._viewer.layers.selection.active}')
        print(f"Type of annotation layer: {type(annotation_layer)}")
        print(f"Type of first label layer: {type(self._viewer.layers[0])}")


if __name__ == "__main__":
    main()