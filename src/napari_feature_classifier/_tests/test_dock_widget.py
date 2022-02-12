import numpy as np
from pathlib import Path
import pandas as pd
from napari_feature_classifier.classifier import Classifier
from napari_feature_classifier.classifier_widgets import (
    initialize_classifier,
    load_classifier,
    ClassifierWidget,
)

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_Classifier_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    test_label_img = np.random.random((100, 100))
    label_layer = viewer.add_labels(test_label_img)

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
    test_label_img = np.random.random((100, 100))
    label_layer = viewer.add_labels(test_label_img)
    test_df_path = Path('test_df.csv')

    # this time, our widget will be a MagicFactory or FunctionGui instance
    my_widget = initialize_classifier(viewer,
                                      label_layer,
                                      test_df_path,
                                      classifier_name='test'
                                      feature_selection=['feature1', 'feature2'],
                                      label_column='label'
                                      )

    # if we "call" this object, it'll execute our function
    my_widget(viewer.layers[0])

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == f"you have selected {layer}\n"

    # "Click" the initialize button
    # TODO: Check that a test.clf file gets generated
    # Check that data gets loaded into classifier correctly


def test_classifier_loading_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()
    test_label_img = np.random.random((100, 100))
    label_layer = viewer.add_labels(test_label_img)
    test_classifier_path = Path('test_classifier.clf')
    test_df_path = Path('test_df.csv')

    # this time, our widget will be a MagicFactory or FunctionGui instance
    my_widget = load_classifier(viewer,
                                label_layer,
                                test_classifier_path,
                                test_df_path)

    # if we "call" this object, it'll execute our function
    my_widget(viewer.layers[0])

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == f"you have selected {layer}\n"
