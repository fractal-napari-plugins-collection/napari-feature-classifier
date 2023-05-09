# napari-feature-classifier

[![License](https://img.shields.io/pypi/l/napari-feature-classifier.svg?color=green)](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-feature-classifier.svg?color=green)](https://pypi.org/project/napari-feature-classifier)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-feature-classifier.svg?color=green)](https://python.org)
[![tests](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/workflows/tests/badge.svg)](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/actions)
[![codecov](https://codecov.io/gh/fractal-napari-plugins-collection/napari-feature-classifier/branch/main/graph/badge.svg)](https://codecov.io/gh/fractal-napari-plugins-collection/napari-feature-classifier)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-feature-classifier)](https://napari-hub.org/plugins/napari-feature-classifier)

An interactive classifier plugin that allows the user to assign objects in a label image to multiple classes and train a classifier to learn those classes based on a feature dataframe.

## Usage
<p align="center"><img src="https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/assets/18033446/1ebf0890-1a7b-4e4b-a21c-88ca8f1dd800" /></p>

To use the napari-feature-classifier, you need to have a label image and corresponding measurements: as a csv file, loaded to layer.features or in an [OME-Zarr Anndata table loaded with another plugin](https://github.com/jluethi/napari-ome-zarr-roi-loader). Your feature measurements need to contain a `label` column that matches the label objects in the label image.
These interactive classification workflows are well suited to visually define cell types, find mitotic cells in images, do quality control by automatically detecting missegmented cells and other tasks where a user can easily assign objects to groups.

#### Prepare the label layer:
- Load your label layer into napari and add the features measurements to layer.features of the corresponding label layer. You can have multiple label layers with their features open at the same time
    - To load features from a CSV file: `Plugins -> napari-feature-classifier -> CSV Feature Loader`, then load the features for the correct label image.
    - To load features from an OME-Zarr file: Get both the label layer into memory as a normal label layer (not a pyramidal label layer, currently untested) and the corresponding features. If your OME-Zarr file is created by [Fractal](https://fractal-analytics-platform.github.io/), you can use [this ROI loader plugin](https://github.com/jluethi/napari-ome-zarr-roi-loader).
    - To load features from anywhere else, load them manually to your label_layer.features
- Your feature table should have 2 columns used for indexing (but stored as normal columns in layer.features):
    - The `label` column to match the object in the label image
    - The `roi_id` column to identify the image you're currently classifying (used when a classifier is trained on multiple label images)


#### Initialize a classifier:
- Start the classifier in napari by going to `Plugins -> napari-feature-classifier -> Initialize a Classifier`  
- Select the features you want to use for the classifier (you need to do the feature selection before initializing. The feature selection can't be changed after initialization anymore). Hold the command key to select multiple features. Feature options are always shown for the features available in the last selected label layer, based on layer.features available features.
- (Optional) Give your classes recognizable names (e.g. Mitotic & Interphase, Cell Type a, b and c etc.)
<img width="1606" alt="Screenshot 2023-05-09 at 11 46 35" src="https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/assets/18033446/452c0d6a-98a3-4e2d-9233-33bfd5bcad19">




#### Classify objects:
<img width="1802" alt="Classifier_annotation" src="https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/assets/18033446/556739b8-972b-4570-9da4-637738fc6a75">

- Make sure you have the label layer selected on which you want to classify
- Select the current class with the radio buttons or by pressing 0, 1, 2, etc.
- Click on label objects in the viewer to assign them to the currently selected class
- Once you have trained enough examples, click "Run Classifier" to run the classifier and have it make a prediction for all objects. Aim for at least a dozen annotations per class, as the classifier divides your annotations 80/20 in training and test sets. 
- Once you get predictions, correct mistakes the classifier made and retrain it to improve its performance.
- You can save the classifier under a different name or in a different location. Define the new output location and then click `Save Classifier` (you need to click the Save Classifier button. Just defining the new output path does not save it yet. But every run of the classifier triggers an autosave)
<img width="1802" alt="Classifier_prediction" src="https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/assets/18033446/69cff600-4585-4a66-9274-d2e7caeb335f">



#### Apply the classifier to additional images:
- You can apply a classifier trained on one image to additional label images. Use `Plugins -> napari-feature-classifier -> Load Classifier`  
- Select the classifier (.clf file with the name you gave above) while already having the label images ready (see `Prepare the label layer` above).
- Click Load Classifier, proceed as above.
<img width="1606" alt="Screenshot 2023-05-09 at 12 01 00" src="https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/assets/18033446/e1143f9f-9729-4f8e-979c-2ab195e0aaca">



#### Export classifier results
- To export the training data and the results of the classifier, define an Export Name (full path to an output file or just a filename ending in .csv) where the results of the classifier shall be saved. It defaults to the layer name for the selected layer in the last directory you chose (or the current working directory if none was chosen so far)
- Click `Export Classifier Result` (Just selecting a filename is not enough, you need to click the export button). This will export the predictions for the currently selected layer.
- The results of the classifier are save in a csv file. The label is an integer of the label object within that image. The prediction column contains predictions of the classifier for all objects (except those that contained NaNs in their feature data) and the annotation column contains the annotations you made (NaN for unclassified objects, -1 for objects you deselected, 1 - 9 for the classes)
<img width="1802" alt="Classifier_prediction" src="https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/assets/18033446/e8f6f7b7-d88b-44f8-b43e-8a2fa81e18d4">


#### Batch mode result export
(To be updated: Create a new notebook to run batch processing, this is for the older version of the classifier)
There is a simple workflow for the classifier in the examples folder:
- Install jupyter-lab (`pip install jupyterlab`)
- Open the notebook in jupyter lab (Type `jupyter-lab` in the terminal when you are in the examples folder)
- Follow the instructions to generate an example dataframe and an example label image
- Use the classifier in napari with this simplified data


#### Initializing the Annotator
You can use the annotation functionality also independently from the classifier
Start the annotator widget by going to `Plugins -> napari-feature-classifier -> Annotator`
Select names for your classes. You can name up to 9 classes. Only classes that you give a name will be created upon initialization.
Then click `Initialize`.

<img width="1411" alt="Screenshot 2023-02-16 at 14 49 38" src="https://user-images.githubusercontent.com/18033446/219384524-9873bd66-270b-4cdd-b913-60d390f6c77a.png">

A annotator widget opens. Use the Radio-Buttons to select what class you're annotating (or keybindings for 1-9 for classes, 0 for deselect).
The annotator will always work on the currently selected label layer. While the annotator is open, you can't edit the labels. Restart napari to allow editing of labels again.

<img width="1411" alt="Screenshot 2023-02-16 at 14 50 00" src="https://user-images.githubusercontent.com/18033446/219384925-b20e4c1a-2eca-4070-8269-902493c5d5ef.png">

The annotations are saved in the `layer.features` table of the corresponding label layer as an `annotations` column.
<img width="1411" alt="Screenshot 2023-02-16 at 15 01 01" src="https://user-images.githubusercontent.com/18033446/219385788-f61bd0a5-fbb6-42d7-81e5-f77ee4d1b4ff.png">


## Installation

This plugin is written for the new napari npe2 plugin engine. Thus, it requires napari >= 0.4.13.
Activate your environment where you have napari installed (or install napari using `pip install "napari[all]"`), then install the classifier plugin:

    pip install napari-feature-classifier

The layer.features dataframes have some issues in napari 0.4.17 (see [here](https://github.com/napari/napari/issues/5617)). They seem to be working again in the nighlty builds. To set up a nightly builds napari env, do the following:

```
conda create -n classifier-dev-napari-main -c "napari/label/nightly" -c conda-forge napari python=3.10 -y
```
    
## Similar napari plugins
If you're looking for other classification approaches, [apoc](https://github.com/haesleinhuepf/apoc) by [Robert Haase](https://github.com/haesleinhuepf) has a pixel classifier in napari and an object classification workflow:  
[napari-accelerated-pixel-and-object-classification (APOC)](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification)  
Alternatively, Clément Cazorla has built [napari-svetlana, a deep learning based classifier](https://www.napari-hub.org/plugins/napari-svetlana)

## Release process
1. Update the version number in src/napari-feature-classifier/__init__.py
2. Update the version in setup.cfg
3. Add a Github release with a new version tag (matching the version set above)
4. Once tests pass, this should automatically be deployed to pypi
5. Wait for conda automation to make a PR for an updated conda release (see https://github.com/conda-forge/napari-feature-classifier-feedstock). This can take 1-2 days. Make sure that PR gets merged.


## Contributing

Contributions are very welcome.

## License

Distributed under the terms of the [BSD-3] license,
"napari-feature-classifier" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

## Contributors
[Joel Lüthi](https://github.com/jluethi) & [Max Hess](https://github.com/MaksHess)

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
