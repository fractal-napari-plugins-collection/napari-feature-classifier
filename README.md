# napari-feature-classifier

[![License](https://img.shields.io/pypi/l/napari-feature-classifier.svg?color=green)](https://github.com/fractal-napari-plugins-collection/napari_feature_classifier/LICENSE)

An interactive classifier plugin that allows the user to assign objects in a label image to multiple classes and train a classifier to learn those classes based on a feature dataframe.

## Installation

Download the repository and manually install it (not on pypi / the napari plugin hub yet)

    git clone https://github.com/fractal-napari-plugins-collection/napari_feature_classifier
    cd napari_feature_classifier
    pip install .



## Usage
#### Initialize a classifier:
- Start the classifier in napari by going to Plugins -> napari-feature-classifier -> Initialize a Classifier  
- Provide a csv file that contains feature measurements and a columns with the integer labels corresponing to the label layer you are using.
- Choose a name (or relative path from the current working directory) for the classifier
- Select the features you want to use for the classifier (can't be changed later in the current implementation). Hold the command key to select multiple features

#### Classify objects:
- Make sure you have the label layer selected on which you want to classify
- Select the current class with the radio buttons or by pressing 0, 1, 2, 3 or 4
- Click on label objects in the viewer to assign them to the currently selected class
- Once you have trained enough examples, click "Run Classifier" to run the classifier and have it make a prediction for all objects. Aim for at least a dozen annotations per class, as the classifier divides your annotations 80/20 in training and test sets. To get good performance readouts, aim for >30 annotations per class.

#### Apply the classifier to additional images:
- You can apply a classifier trained on one image to additional label images. Use Plugins -> napari-feature-classifier -> Load Classifier  
- Select the classifier (.clf file with the name you gave above) and a dataframe containing the same features as the past images.
- Click Load Classifier, proceed as above.



## Contributing

Contributions are very welcome.

## License

Distributed under the terms of the [BSD-3] license,
"napari-feature-classifier" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

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
