"""Older ``.clf`` files must keep loading after the core split.

A ``.clf`` is a pickled ``Classifier``. Files written before the class moved
to :mod:`feature_classifier_core` embed the old path
``napari_feature_classifier.classifier.Classifier``; the re-export shim in
:mod:`napari_feature_classifier.classifier` maps that onto the moved class so
existing files keep unpickling. This guards against the shim regressing.
"""

from pathlib import Path

# Import through the shim on purpose: this is the path legacy pickles reference.
from napari_feature_classifier.classifier import Classifier

# A real classifier saved by an older plugin version (pickled against
# napari_feature_classifier.classifier, pandas < the current major).
TEST_CLF = Path(
    "src/napari_feature_classifier/sample_data/test_labels_classifier.clf"
)


def test_legacy_clf_loads_through_shim():
    clf = Classifier.load(str(TEST_CLF))

    assert isinstance(clf, Classifier)
    # Reconstructed as the moved class, reached via the old pickled path.
    assert type(clf).__module__ == "feature_classifier_core.classifier"
    assert clf.get_feature_names() == ["feature1", "feature2"]
    assert clf.get_class_names() == ["Class_1", "Class_2", "Class_3"]
    assert clf.get_estimator() is not None
