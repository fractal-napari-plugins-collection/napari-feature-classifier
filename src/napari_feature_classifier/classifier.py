"""Back-compatibility shim.

The classifier is part of the headless :mod:`feature_classifier_core` package.
This module re-exports it so that ``from napari_feature_classifier.classifier
import Classifier`` keeps working, and that ``.clf`` files which
referencing ``napari_feature_classifier.classifier.Classifier`` still unpickle.

New code should import from :mod:`feature_classifier_core` directly.
"""

from feature_classifier_core.classifier import (  # noqa: F401
    BUNDLE_FORMAT_VERSION,
    Classifier,
    get_input_internal_and_predict_schemas,
    get_normalized_hash_column,
    get_random_object_id,
    hash_single_object_id,
    join_index_columns,
)

__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "Classifier",
    "get_input_internal_and_predict_schemas",
    "get_normalized_hash_column",
    "get_random_object_id",
    "hash_single_object_id",
    "join_index_columns",
]
