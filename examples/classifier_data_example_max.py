from pathlib import Path

import h5py
import napari
import pandas as pd

from src.napari_feature_classifier.classifier import Classifier
from src.napari_feature_classifier.classifier_widget import ClassifierWidget
from src.napari_feature_classifier.data import ClassifierData


def main():
    # fn = r"Z:\hmax\Zebrafish\20211119_cyclerDATA_compressed\20211119_ABpanelTestset_3pair_3.5h_1_s6.h5"
    # with h5py.File(fn) as f:
    #     sytox = f['ch_01/1'][:, 512:1024, 512:1024]
    #     lbl = f['lbl_nuc'][:, 512:1024, 512:1024]
    # viewer = napari.Viewer()
    # viewer.add_image(sytox, scale=(1.0, 0.65, 0.65))
    # lbl_layer = viewer.add_labels(lbl, scale=(1.0, 0.65, 0.65))

    feature_path = r"C:\Users\hessm\Documents\Programming\Python\classifier_demo\featuresNucs_3pair_3.5h_1_s6.csv"
    training_feature_names = ['Roundness', 'NumberOfPixels']

    data = ClassifierData.from_path(feature_path, training_feature_names=training_feature_names,
                                    index_columns=['filename_prefix', 'Label'])

    print(data.features.head())
    print()
    print(data.validate_index(data.features).head())

    # widget = ClassifierWidget(lbl_layer, feature_path, 'mito', training_feature_names, 'Label', viewer)

    # viewer.show(block=True)


if __name__ == '__main__':
    main()
