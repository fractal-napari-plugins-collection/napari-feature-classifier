""" Tests for 3 core dock widgets to see if their initialization generates errors"""
import numpy as np
from pathlib import Path
import pandas as pd
import numpy as np
from napari_feature_classifier.classifier import Classifier
from napari_feature_classifier.classifier_widgets import (
    initialize_classifier,
    load_classifier,
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
def test_classifier_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)

    # Load test data
    test_df_path = Path('test_df.csv')
    test_features_df = pd.read_csv(test_df_path)
    test_features_df["path"] = test_df_path

    # Create a classifier
    clf = Classifier(name='classifier_test',
                     features=test_features_df,
                     training_features=['feature1', 'feature2'],
                     method='rfc',
                     directory=Path('.'),
                     index_columns=("path", label_column),
                    )
    test_df_path = Path('test_df.csv')

    # create our widget, passing in the viewer
    my_widget = ClassifierWidget(clf, label_layer, test_df_path, viewer)

    # call our widget method
    my_widget._on_click()

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == "napari has 1 layers\n"


def test_classifier_initialization_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)
    test_df_path = Path('test_df.csv')

    # this time, our widget will be a MagicFactory or FunctionGui instance
    my_widget = initialize_classifier(viewer,
                                      label_layer,
                                      test_df_path,
                                      classifier_name='test',
                                      feature_selection=['feature1', 'feature2'],
                                      label_column='label'
                                      )

    # if we "call" this object, it'll execute our function
    my_widget(viewer.layers[0]) # pylint: disable-msg=E1102

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == f"you have selected {layer}\n"

    # "Click" the initialize button
    # TODO: Check that a test.clf file gets generated
    # Check that data gets loaded into classifier correctly


def test_classifier_loading_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    label_layer = viewer.add_labels(lbl_img_np)
    test_classifier_path = Path('test_classifier.clf')
    test_df_path = Path('test_df.csv')

    # this time, our widget will be a MagicFactory or FunctionGui instance
    my_widget = load_classifier(viewer,
                                label_layer,
                                test_classifier_path,
                                test_df_path)

    # if we "call" this object, it'll execute our function
    my_widget(viewer.layers[0]) # pylint: disable-msg=E1102

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == f"you have selected {layer}\n"
