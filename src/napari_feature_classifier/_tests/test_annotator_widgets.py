""" Tests for annotator widget initialization"""
import imageio
from pathlib import Path
from napari_feature_classifier.annotator_init_widget import (
    InitializeLabelAnnotatorWidget,
)
from napari_feature_classifier.annotator_widget import (
    LabelAnnotator,
    get_class_selection,
)

lbl_img_np = imageio.v2.imread(
    Path("src/napari_feature_classifier/sample_data/test_labels.tif")
)


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_annotator_widgets(make_napari_viewer):
    """
    Tests if the AnnotatorInit & Annotator widget can be initialized
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)

    # Start init widget
    init_widget = InitializeLabelAnnotatorWidget(viewer)
    init_widget.initialize_annotator()

    # Start the annotator widget
    annotator_widget = LabelAnnotator(viewer)

    # call our widget method
    # my_widget._on_click()
    annotator_widget._init_annotation(label_layer)

    # TODO: Test toggle_label


# TODO: Test saving annotations


def test_custom_class_selection(make_napari_viewer):
    """
    Tests the custom class selection
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_labels(lbl_img_np)

    class_names = ["Class Test", "Class XYZ", "12345"]
    # Start the annotator widget with a list of named classes
    LabelAnnotator(viewer, ClassSelection=get_class_selection(class_names=class_names))


def test_numbered_class_selection(make_napari_viewer):
    """
    Tests the custom class selection
    """
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    viewer.add_labels(lbl_img_np)
    # Start the annotator widget with a given number of classes
    LabelAnnotator(viewer, ClassSelection=get_class_selection(n_classes=8))
