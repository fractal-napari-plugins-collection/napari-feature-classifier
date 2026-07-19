# feature-classifier-core

Headless core of [napari-feature-classifier](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier).

It provides the `Classifier` (training, prediction and portable, versioned
model bundles) without **napari, Qt or magicgui dependencies** for use in
headless contexts. Install the full
`napari-feature-classifier` package for the interactive napari GUI.

```python
import pandas as pd
from feature_classifier_core import Classifier

# 1. Upstream: Initialize, train and export the classifier
annotated_df = pd.DataFrame(
    {
        "roi_id": ["img1"] * 6,
        "label": [1, 2, 3, 4, 5, 6],
        "feature_1": [0.1, 0.2, 0.15, 0.9, 0.8, 0.95],
        "feature_2": [1.0, 1.1, 0.9, 5.0, 5.2, 4.8],
        "annotations": [1, 1, 1, 2, 2, 2],
    }
)

clf = Classifier(feature_names=["feature_1", "feature_2"], class_names=["A", "B"])
clf.add_features(annotated_df)
clf.train()
clf.export_bundle("model.joblib")

# 2. Downstream: Load the classifier and predict labels
features_df = pd.DataFrame(
    {
        "roi_id": ["img2", "img2"],
        "label": [1, 2],
        "feature_1": [0.12, 0.88],
        "feature_2": [1.05, 5.1],
    }
)
clf = Classifier.load("model.joblib")
predictions = clf.predict(features_df)
```

The neutral bundle is a plain dict
`{format_version, estimator, feature_names, class_names}` that downstream tools
can consume with `joblib` + `scikit-learn` without importing this package.

```python
import joblib

bundle = joblib.load("model.joblib")
estimator = bundle["estimator"]          # a fitted scikit-learn estimator
feature_names = bundle["feature_names"]
predictions = estimator.predict(features_df[feature_names])
```
