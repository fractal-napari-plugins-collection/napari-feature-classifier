from typing import Optional
import numpy as np
import pandas as pd
from magicgui import magic_factory, widgets, magicgui
from napari_plugin_engine import napari_hook_implementation
import napari
from pathlib import Path


def main():
    # print(initialize_annotator())
    viewer = napari.Viewer()
    # viewer.add_labels(np.zeros((10, 10), dtype=int))
    # napari_experimental_provide_dock_widget()
    viewer.add_labels(np.zeros((10, 10), dtype=np.uint16))
    # viewer.add_label
    # threshold_widget = threshold()
    viewer.window.add_dock_widget(initialize_annotator)
    viewer.show(block=True)
    # napari_experimental_provide_dock_widget()


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
    widget.label_layer.value.features['annotation'] = pd.Series(dtype=pd.Int64Dtype)
    # TODO: create layer


@magicgui(classes={'widget_type': 'RadioButtons', 'choices': ['Clear', 'Class 1', 'Class 2']},
          # widget_init=_initialize_annotation_stuff,
          )
def initialize_annotator(
        label_layer: "napari.layers.Labels",
        classes: list[str],
        output_path: Optional[Path] = Path('.') / 'annotation.csv',
):

    return label_layer





if __name__ == '__main__':
    main()
