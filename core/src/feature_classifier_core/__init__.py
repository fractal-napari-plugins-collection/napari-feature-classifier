"""Headless core of the feature classifier.

Contains the :class:`Classifier` (training, prediction, and neutral model
bundles) and its helper functions, with no napari/Qt/magicgui dependency. The
interactive napari plugin ``napari-feature-classifier`` builds its widgets on
top of this package.
"""

try:
    from feature_classifier_core._version import __version__
except ImportError:
    __version__ = "unknown"

from feature_classifier_core.classifier import (
    BUNDLE_FORMAT_VERSION,
    Classifier,
    check_bundle_version,
    get_input_internal_and_predict_schemas,
    get_normalized_hash_column,
    get_random_object_id,
    hash_single_object_id,
    join_index_columns,
    load_bundle,
)

__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "Classifier",
    "check_bundle_version",
    "get_input_internal_and_predict_schemas",
    "get_normalized_hash_column",
    "get_random_object_id",
    "hash_single_object_id",
    "join_index_columns",
    "load_bundle",
    "__version__",
]
