"""Tests for the live annotation count display in ClassifierRunContainer."""

import numpy as np
import pytest

from napari_feature_classifier.classifier_widget import ClassifierRunContainer
from napari_feature_classifier.feature_loader_widget import make_features

# ---------------------------------------------------------------------------
# Shared constants & helpers
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["feature_0", "feature_1", "feature_2"]
CLASS_NAMES = ["Class_1", "Class_2"]
N_LABELS = 10


def make_label_layer(viewer, name="Labels", roi_id="site1"):
    img = np.zeros((10, 10), dtype=np.int32)
    for i in range(N_LABELS):
        img[i // 10, i % 10] = i + 1
    layer = viewer.add_labels(img, name=name)
    layer.features = make_features(
        labels=list(range(1, N_LABELS + 1)), roi_id=roi_id, n_features=3
    )
    return layer


def make_run_container(viewer):
    return ClassifierRunContainer(
        viewer, class_names=CLASS_NAMES, feature_names=FEATURE_NAMES
    )


def parse_counts(count_str: str) -> dict[str, int]:
    """Parse "Class_1: 3\nClass_2: 2" → {"Class_1": 3, "Class_2": 2}."""
    if not count_str:
        return {}
    return {k: int(v) for k, v in (line.split(": ") for line in count_str.splitlines())}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def viewer(make_napari_viewer):
    return make_napari_viewer()


# ---------------------------------------------------------------------------
# Empty / no annotations
# ---------------------------------------------------------------------------


def test_count_all_zeros_when_no_annotations_column(viewer):
    """Layer with no annotations column → all classes show 0."""
    make_label_layer(viewer)
    container = make_run_container(viewer)
    counts = parse_counts(container._get_annotation_counts())
    assert counts == {"Class_1": 0, "Class_2": 0}


def test_count_all_zeros_when_all_annotations_are_nan(viewer):
    """Layer whose annotations column is entirely NaN → all zeros."""
    layer = make_label_layer(viewer)
    layer.features["annotations"] = float("nan")
    container = make_run_container(viewer)
    counts = parse_counts(container._get_annotation_counts())
    assert counts == {"Class_1": 0, "Class_2": 0}


# ---------------------------------------------------------------------------
# Single open layer
# ---------------------------------------------------------------------------


def test_count_single_layer_basic(viewer):
    """Correct per-class counts from a single open layer."""
    layer = make_label_layer(viewer)
    layer.features.loc[0, "annotations"] = 1.0
    layer.features.loc[1, "annotations"] = 1.0
    layer.features.loc[2, "annotations"] = 1.0
    layer.features.loc[3, "annotations"] = 2.0
    layer.features.loc[4, "annotations"] = 2.0
    container = make_run_container(viewer)
    counts = parse_counts(container._get_annotation_counts())
    assert counts == {"Class_1": 3, "Class_2": 2}


def test_count_updates_after_annotation_write(viewer):
    """Writing a new annotation and calling _update_count_label refreshes the display."""
    layer = make_label_layer(viewer)
    layer.features.loc[0, "annotations"] = 1.0
    container = make_run_container(viewer)
    assert parse_counts(container._count_label.value) == {"Class_1": 1, "Class_2": 0}

    layer.features.loc[1, "annotations"] = 1.0
    container._update_count_label()
    assert parse_counts(container._count_label.value) == {"Class_1": 2, "Class_2": 0}


def test_count_label_updated_via_callback(viewer):
    """The callback wired into LabelAnnotator updates _count_label.value."""
    layer = make_label_layer(viewer)
    container = make_run_container(viewer)
    initial = container._count_label.value

    layer.features.loc[0, "annotations"] = 2.0
    # Fire callbacks the same way toggle_label() does
    for cb in container._annotator._annotation_callbacks:
        cb()

    assert container._count_label.value != initial
    assert parse_counts(container._count_label.value)["Class_2"] == 1


# ---------------------------------------------------------------------------
# Two open label layers
# ---------------------------------------------------------------------------


def test_count_aggregates_two_open_layers(viewer):
    """Counts are summed across all valid open layers."""
    layer1 = make_label_layer(viewer, name="Labels1", roi_id="site1")
    layer2 = make_label_layer(viewer, name="Labels2", roi_id="site2")

    layer1.features.loc[0, "annotations"] = 1.0
    layer1.features.loc[1, "annotations"] = 1.0  # 2 × Class_1 on layer1
    layer2.features.loc[0, "annotations"] = 1.0  # 1 × Class_1 on layer2
    layer2.features.loc[1, "annotations"] = 2.0
    layer2.features.loc[2, "annotations"] = 2.0
    layer2.features.loc[3, "annotations"] = 2.0  # 3 × Class_2 on layer2

    container = make_run_container(viewer)
    counts = parse_counts(container._get_annotation_counts())
    assert counts == {"Class_1": 3, "Class_2": 3}


# ---------------------------------------------------------------------------
# Closed image (historical data in classifier._data)
# ---------------------------------------------------------------------------


def test_count_includes_closed_image(viewer):
    """After a layer is removed, its annotations persist via classifier._data."""
    layer1 = make_label_layer(viewer, name="Labels1", roi_id="site1")
    layer1.features.loc[0, "annotations"] = 1.0
    layer1.features.loc[1, "annotations"] = 2.0

    container = make_run_container(viewer)
    # Push annotations into classifier._data
    container._runner.add_features_to_classifier()

    # Remove the layer — simulates closing the image
    viewer.layers.remove(layer1)

    counts = parse_counts(container._get_annotation_counts())
    assert counts == {"Class_1": 1, "Class_2": 1}


def test_count_live_overrides_stale_history(viewer):
    """Open layer takes precedence over stale historical data for the same roi_id."""
    layer = make_label_layer(viewer, name="Labels", roi_id="site1")
    layer.features.loc[0, "annotations"] = 1.0
    layer.features.loc[1, "annotations"] = 1.0  # 2 × Class_1 in history

    container = make_run_container(viewer)
    container._runner.add_features_to_classifier()  # persist to _data

    # Now change live annotations on the same layer (same roi_id)
    layer.features["annotations"] = float("nan")  # clear
    layer.features.loc[0, "annotations"] = 1.0
    layer.features.loc[1, "annotations"] = 1.0
    layer.features.loc[2, "annotations"] = 1.0  # 3 × Class_1 live
    layer.features.loc[3, "annotations"] = 2.0  # 1 × Class_2 live

    counts = parse_counts(container._get_annotation_counts())
    # Live data should win — NOT 2 (history) + 3 (live) = 5
    assert counts == {"Class_1": 3, "Class_2": 1}


def test_count_closed_plus_open(viewer):
    """Historical closed-image annotations are combined with live open-layer annotations."""
    layer1 = make_label_layer(viewer, name="Labels1", roi_id="site1")
    layer1.features.loc[0, "annotations"] = 1.0
    layer1.features.loc[1, "annotations"] = 1.0  # 2 × Class_1 from site1

    container = make_run_container(viewer)
    container._runner.add_features_to_classifier()

    # Remove layer1, add layer2
    viewer.layers.remove(layer1)
    layer2 = make_label_layer(viewer, name="Labels2", roi_id="site2")
    layer2.features.loc[0, "annotations"] = 2.0  # 1 × Class_2 from site2

    counts = parse_counts(container._get_annotation_counts())
    assert counts == {"Class_1": 2, "Class_2": 1}


# ---------------------------------------------------------------------------
# After run()
# ---------------------------------------------------------------------------


def test_count_label_still_correct_after_run(viewer, tmp_path, monkeypatch):
    """_count_label.value is refreshed correctly after run() completes."""
    monkeypatch.chdir(tmp_path)
    layer = make_label_layer(viewer)
    # 4 × Class_1 and 4 × Class_2: enough for the hash-based train/test split
    for idx in range(4):
        layer.features.loc[idx, "annotations"] = 1.0
    for idx in range(4, 8):
        layer.features.loc[idx, "annotations"] = 2.0

    container = make_run_container(viewer)
    container.run()

    counts = parse_counts(container._count_label.value)
    assert counts["Class_1"] == 4
    assert counts["Class_2"] == 4


# ---------------------------------------------------------------------------
# selection_changed hook
# ---------------------------------------------------------------------------


def test_count_updates_on_layer_selection_change(viewer):
    """Switching the active layer triggers _update_count_label via selection_changed."""
    layer1 = make_label_layer(viewer, name="Labels1", roi_id="site1")
    layer2 = make_label_layer(viewer, name="Labels2", roi_id="site2")

    layer1.features.loc[0, "annotations"] = 1.0
    layer2.features.loc[0, "annotations"] = 2.0

    container = make_run_container(viewer)

    # Switch active layer — selection_changed should fire _update_count_label
    viewer.layers.selection.active = layer1
    counts_after_switch = parse_counts(container._count_label.value)

    # Both layers are open, so both annotations should be visible
    assert counts_after_switch["Class_1"] == 1
    assert counts_after_switch["Class_2"] == 1
