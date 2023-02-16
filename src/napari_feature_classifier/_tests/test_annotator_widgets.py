""" Tests for 3 core dock widgets to see if their initialization generates errors"""
from pathlib import Path
import numpy as np
import pandas as pd
from napari_feature_classifier.annotator_init_widget import InitializeLabelAnnotator
from napari_feature_classifier.annotator_widget import (
    LabelAnnotator, 
    get_class_selection
)

# Define a simple test label image for all widgets
shape = (1, 50, 50)
lbl_img_np = np.zeros(shape).astype('uint16')
lbl_img_np[0, 5:10, 5:10] = 1
lbl_img_np[0, 15:20, 5:10] = 2
lbl_img_np[0, 25:30, 5:10] = 3
lbl_img_np[0, 5:10, 15:20] = 4
lbl_img_np[0, 15:20, 15:20] = 5
lbl_img_np[0, 25:30, 15:20] = 6
lbl_img_np[0, 35:40, 15:20] = 7
lbl_img_np[0, 35:40, 25:30] = 8
lbl_img_np[0, 5:10, 35:40] = 9
lbl_img_np[0, 25:30, 25:30] = 10
lbl_img_np[0, 25:30, 35:40] = 11
lbl_img_np[0, 5:10, 25:30] = 12
lbl_img_np[0, 15:20, 25:30] = 13
lbl_img_np[0, 15:20, 35:40] = 14
lbl_img_np[0, 35:40, 5:10] = 15
lbl_img_np[0, 35:40, 35:40] = 16


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_annotator_widgets(make_napari_viewer, capsys):
    """
    Tests if the AnnotatorInit & Annotator widget can be initialized
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)

    # Start init widget
    _ = InitializeLabelAnnotator(viewer)

    # Start the annotator widget
    annotator_widget = LabelAnnotator(viewer)

    # call our widget method
    #my_widget._on_click()
    annotator_widget._init_annotation(label_layer)


def test_custom_class_selection(make_napari_viewer, capsys):
    """
    Tests the custom class selection
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_labels(lbl_img_np)

    class_names = ['Class Test', 'Class XYZ', '12345']
    # Start the annotator widget with a list of named classes
    LabelAnnotator(
        viewer,
        ClassSelection = get_class_selection(class_names = class_names)
    )

def test_numbered_class_selection(make_napari_viewer, capsys):
    """
    Tests the custom class selection
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_labels(lbl_img_np)
    # Start the annotator widget with a given number of classes
    LabelAnnotator(
        viewer,
        ClassSelection = get_class_selection(n_classes = 8)
    )
