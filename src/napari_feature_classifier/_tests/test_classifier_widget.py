""" Tests for classifier widget initialization"""
import numpy as np
from napari_feature_classifier.classifier_widget import (
    ClassifierWidget,
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
def test_classifier_widgets(make_napari_viewer):
    """
    Tests if the main widget launches
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    _ = viewer.add_labels(lbl_img_np)

    # Start init widget
    classifier_widget = ClassifierWidget(viewer)

    classifier_widget.initialize_run_widget()
