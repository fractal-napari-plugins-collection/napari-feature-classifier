# napari-feature-classifier

[![License](https://img.shields.io/pypi/l/napari-feature-classifier.svg?color=green)](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-feature-classifier.svg?color=green)](https://pypi.org/project/napari-feature-classifier)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-feature-classifier.svg?color=green)](https://python.org)
[![tests](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/workflows/tests/badge.svg)](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/actions)
[![codecov](https://codecov.io/gh/fractal-napari-plugins-collection/napari-feature-classifier/branch/main/graph/badge.svg)](https://codecov.io/gh/fractal-napari-plugins-collection/napari-feature-classifier)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-feature-classifier)](https://napari-hub.org/plugins/napari-feature-classifier)

An interactive classifier plugin that allows the user to assign objects in a label image to multiple classes and train a classifier to learn those classes based on a feature dataframe.


## Usage
<p align="center"><img src="https://user-images.githubusercontent.com/18033446/153727595-60380204-f299-485f-b762-d2030b75e7d3.gif" /></p>
To use the napari-feature-classifier, you need to have a label image and a csv file containing measurements that correspond to the object in the label image. The csv file needs to contain a column with integer values corresponding to the label values in the label image.
These interactive classification workflows are well suited to visually define cell types, find mitotic cells in images, do quality control by automatically detecting missegmented cells and other tasks where a user can easily assign objects to groups.

#### Initialize a classifier:
- Start the classifier in napari by going to `Plugins -> napari-feature-classifier -> Initialize a Classifier`  
- Provide a csv file that contains feature measurements and a column with the integer labels corresponding to the label layer you are using.
- Choose a name (or relative path from the current working directory) for the classifier. The classifier is initially saved in the current working directory (you can change this later on).
- Select the features you want to use for the classifier (you need to do the feature selection before initializing. The feature selection can't be changed after initialization anymore). Hold the command key to select multiple features.
<img width="1831" alt="Initialize Classifier" src="https://user-images.githubusercontent.com/18033446/153727784-d7b7d44b-a7b1-479f-a4af-34e0e280c8d6.png">


#### Classify objects:
- Make sure you have the label layer selected on which you want to classify
- Select the current class with the radio buttons or by pressing 0, 1, 2, 3 or 4
- Click on label objects in the viewer to assign them to the currently selected class
- While you need to have the label layer active to select, sometimes you want to focus on the intensity images. You can press `v` (or manually change the opacity of the label layer) to focus on the intensity images.
- Once you have trained enough examples, click "Run Classifier" (or press `t`) to run the classifier and have it make a prediction for all objects. Aim for at least a dozen annotations per class, as the classifier divides your annotations 80/20 in training and test sets. To get good performance readouts, aim for >30 annotations per class.
- Once you get predictions, correct mistakes the classifier made and retrain it to improve its performance.
- You can save the classifier under a different name (to move it to a new folder or to have a slightly different version of the classifier - but careful, it autosaves whenever you run it). Define the new output location and then click `Save Classifier` (you need to click the Save Classifier button. Just defining the new output path does not save it yet)
<img width="1831" alt="trainClassifier" src="https://user-images.githubusercontent.com/18033446/153727960-daae2955-4368-4081-88da-1a1cdbda6e69.png">


#### Apply the classifier to additional images:
- You can apply a classifier trained on one image to additional label images. Use `Plugins -> napari-feature-classifier -> Load Classifier`  
- Select the classifier (.clf file with the name you gave above) and a csv file containing the same features as the past images.
- Click Load Classifier, proceed as above.
<img width="1831" alt="LoadClassifier" src="https://user-images.githubusercontent.com/18033446/153728100-dd60918d-c9a4-4de8-8f0e-8fd8c6a51700.png">


#### Export classifier results
- To export the training data and the results of the classifier, define an Export Name (full path to an output file or just a filename ending in .csv) where the results of the classifier shall be saved
- Click `Export Classifier Result` (Just selecting a filename is not enough, you need to click the export button)
- The results of the classifier are save in a csv file. The first two columns are index columns: path describes the Feature Path used (and allows you to understand which image / feature dataframe a result is from) and label is an integer of the label object within that image. The predict column contains predictions of the classifier for all objects (except those that contained NaNs in their feature data) and the train column contains the annotations you made (0 for unclassified objects, 1, 2, 3 or 4 for the classes)
![DataStructure](https://user-images.githubusercontent.com/18033446/153728461-d685987d-e1a9-46ff-834b-073008252ccb.png)


There is a simple workflow for the classifier in the examples folder:
- Install jupyter-lab (`pip install jupyterlab`)
- Open the notebook in jupyter lab (Type `jupyter-lab` in the terminal when you are in the examples folder)
- Follow the instructions to generate an example dataframe and an example label image
- Use the classifier in napari with this simplified data

## Refactored Classifier
We're currently in the process of refactoring the classifier code to make it more modular. As a first step, we have created a separate Annotator widget that is already available in version 0.0.2 of the classifier. The current classifier doesn't make use of these annotations yet, so only use the new annotator widget if you need annotation only. We are refactoring the classifier to also work with this and will release the refactored classifier later.

#### Initializing the new Annotator
Start the annotator widget by going to `Plugins -> napari-feature-classifier -> Annotator`
Select names for your classes. You can name up to 9 classes. Only classes that you give a name will be created upon initialization.
Then click `Initialize`.

<img width="1411" alt="Screenshot 2023-02-16 at 14 49 38" src="https://user-images.githubusercontent.com/18033446/219384524-9873bd66-270b-4cdd-b913-60d390f6c77a.png">

A new annotator widget opens. Use the Radio-Buttons to select what class you're annotating (or keybindings for 1-9 for classes, 0 for deselect).
The annotator will always work on the currently selected label layer. While the annotator is open, you can't edit the labels. Restart napari to allow editing of labels again.

<img width="1411" alt="Screenshot 2023-02-16 at 14 50 00" src="https://user-images.githubusercontent.com/18033446/219384925-b20e4c1a-2eca-4070-8269-902493c5d5ef.png">

The annotations are saved in the `layer.features` table of the corresponding label layer as an `annotations` column.
<img width="1411" alt="Screenshot 2023-02-16 at 15 01 01" src="https://user-images.githubusercontent.com/18033446/219385788-f61bd0a5-fbb6-42d7-81e5-f77ee4d1b4ff.png">


## Installation

This plugin is written for the new napari npe2 plugin engine. Thus, it requires napari >= 0.4.13.
Activate your environment where you have napari installed (or install napari using `pip install "napari[all]"`), then install the classifier plugin:

    pip install napari-feature-classifier
    
## Similar napari plugins
If you're looking for other classification approaches, [apoc](https://github.com/haesleinhuepf/apoc) by [Robert Haase](https://github.com/haesleinhuepf) has a pixel classifier in napari and an object classification workflow:  
[napari-accelerated-pixel-and-object-classification (APOC)](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification)
Alternatively, Clément Cazorla has built [napari-svetlana, a deep learning based classifier](https://www.napari-hub.org/plugins/napari-svetlana)


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
