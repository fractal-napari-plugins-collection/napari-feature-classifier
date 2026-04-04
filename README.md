# napari-feature-classifier

[![License](https://img.shields.io/pypi/l/napari-feature-classifier.svg?color=green)](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-feature-classifier.svg?color=green)](https://pypi.org/project/napari-feature-classifier)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-feature-classifier.svg?color=green)](https://python.org)
[![tests](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/workflows/tests/badge.svg)](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/actions)
[![codecov](https://codecov.io/gh/fractal-napari-plugins-collection/napari-feature-classifier/branch/main/graph/badge.svg)](https://codecov.io/gh/fractal-napari-plugins-collection/napari-feature-classifier)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-feature-classifier)](https://napari-hub.org/plugins/napari-feature-classifier)

An interactive classifier plugin for napari that lets you annotate objects in a label image and train a random forest classifier to generalize those annotations across all objects — without leaving the viewer.

![napari-feature-classifier](https://github.com/user-attachments/assets/e1a3156a-bc78-442e-9294-f81aba503ee4)

## When to use this

This plugin is designed for tasks where visual inspection defines the ground truth:
- Classifying cell types (e.g. mitotic vs. interphase cells)
- Quality control (flagging mis-segmented objects)
- Any labeling task where you can recognize the classes by eye but want to scale it to thousands of objects automatically

You need: a **label image** and a **feature table** — measurements per object (area, intensity, shape descriptors, etc.) stored in `layer.features`. The classifier learns from your manual annotations and applies those patterns to every object.

## Usage

### 1. Prepare your label layer

Load your label image into napari and attach feature measurements to `layer.features` of that layer. You can have multiple label layers open at once — the classifier handles them all.

Your feature table must have:
- A `label` column matching the integer labels in the image
- A `roi_id` column identifying which image each row belongs to (used when training on multiple images)

**Ways to load features:**
- From an OME-Zarr file: use [napari-ome-zarr-navigator](https://github.com/fractal-napari-plugins-collection/napari-ome-zarr-navigator), which handles correct loading of both the label image and the feature table and populates the `roi_id` column automatically.
- From a CSV file: `Plugins → napari-feature-classifier → CSV Feature Loader`, then select the label layer and point to the CSV.
- Programmatically: `label_layer.features = your_dataframe`

### 2. Initialize a classifier

Go to `Plugins → napari-feature-classifier → Initialize a Classifier`.

<img width="1694" height="1088" alt="classifier_init" src="https://github.com/user-attachments/assets/05a2495a-b7fd-40be-bd9c-607177c0aa68" />


- Select the features to use for training. Hold Cmd/Ctrl to select multiple. The feature list reflects the currently selected label layer.
- Name your classes (e.g. "Mitotic", "Interphase"). Classes without a name won't be created.
- Click **Initialize**.

> Feature selection is fixed after initialization. If you want different features, start a new classifier.

### 3. Annotate and train

<img width="1694" height="1088" alt="classifier_annotation" src="https://github.com/user-attachments/assets/25a9c90a-0d58-4fc2-92df-9006cdae00aa" />


- Select your label layer in the napari layer list.
- Pick a class using the panel buttons or keyboard shortcuts (keys **1–9** for classes, **0** to deselect).
- Click on label objects in the viewer to annotate them.
- The live count display shows how many objects you've annotated per class across all open images.
- Once you have at least a handful of examples per class (aim for 10+), click **Run Classifier**.

The classifier splits your annotations 80/20 into training and test sets, trains a random forest, and applies it to all objects. Predictions appear as a color-coded **Predictions** layer.

<img width="1694" height="1088" alt="classifier-predict" src="https://github.com/user-attachments/assets/273d696f-669f-4d01-88e4-e74e204a066d" />

Correct mistakes the classifier made and click Run Classifier again to improve it. Iterative annotation is the intended workflow.

### 4. Save and reload

After each run, the classifier auto-saves to a `.clf` file named after the label layer (in the current working directory). To save to a different location: expand **▶ Saving & Export** and click **Save Classifier As…**.

To resume work or apply a trained classifier to new images:
`Plugins → napari-feature-classifier → Load Classifier`

Select the `.clf` file, make sure your label layers with features are already open, and click **Load Classifier**. 

### 5. Export results

Expand **▶ Saving & Export** and click **Export Results As…** to save predictions for the currently selected layer as a CSV file.

The exported CSV contains:
- `label` — integer object ID
- `prediction` — classifier prediction (1–N for each class; NaN for objects with missing features)
- `annotations` — your manual annotations (NaN = not annotated, −1 = explicitly deselected, 1–N = class)
- One column per annotation class name

<img width="317" height="137" alt="classifier-save" src="https://github.com/user-attachments/assets/62236f0d-7cc5-4760-8158-efe65e52109a" />


### 6. Standalone annotator

You can use the annotation tool independently from the classifier:
`Plugins → napari-feature-classifier → Annotator`

Name up to 9 classes, click **Initialize**, then annotate as above. Annotations are stored in `layer.features["annotations"]` and can be saved to CSV via **▶ Save Annotations**.

<img width="322" height="239" alt="annotator" src="https://github.com/user-attachments/assets/05c3bebc-3d9c-4a78-bfde-b821e8f565cc" />

### Batch / scripted use

Classifiers can be applied programmatically without the napari UI. See [examples/simple_classifier_example.ipynb](examples/simple_classifier_example.ipynb) for a worked example.

## Installation

Requires Python ≥ 3.10 and napari ≥ 0.6.0.

We recommend installing into a dedicated environment to avoid dependency conflicts:

```bash
# With conda
conda create -n napari-feature-classifier -c conda-forge napari python=3.12 -y
conda activate napari-feature-classifier
pip install napari-feature-classifier
```

Or with [pixi](https://pixi.sh):

```bash
pixi init my-project
pixi add napari napari-feature-classifier
pixi run napari
```

Or into an existing environment:

```bash
pip install napari-feature-classifier
```

## Similar napari plugins

- [napari-convpaint](https://github.com/guiwitz/napari-convpaint) — deep feature-based pixel and object classifier by Guillaume Witz
- [napari-accelerated-pixel-and-object-classification (APOC)](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification) — pixel and object classifier by Robert Haase
- [napari-svetlana](https://www.napari-hub.org/plugins/napari-svetlana) — deep learning based classifier by Clément Cazorla

## Release process

1. Tag a release on GitHub with the new version number (e.g. `v0.3.2`). The version is set automatically from the git tag via `hatch-vcs`.
2. Once CI tests pass, the package is automatically deployed to PyPI.
3. A conda-forge PR will be opened automatically within 1–2 days — review and merge it at [napari-feature-classifier-feedstock](https://github.com/conda-forge/napari-feature-classifier-feedstock).

## Contributing

Contributions are very welcome. Please open an issue to discuss significant changes before starting work.

## License

Distributed under the terms of the [BSD-3-Clause](LICENSE) license.

## Issues

If you encounter any problems, please [file an issue](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier/issues) along with a detailed description.

## Contributors

[Joel Lüthi](https://github.com/jluethi) & [Max Hess](https://github.com/MaksHess)
