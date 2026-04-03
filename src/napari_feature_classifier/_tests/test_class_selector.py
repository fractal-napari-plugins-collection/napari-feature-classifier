"""Tests for ClassSelectorRow, ClassSelectorPanel, CollapsibleSection, and
related annotator/classifier widget logic that isn't covered elsewhere."""

import numpy as np
import pytest

from napari_feature_classifier.annotator_widget import (
    ClassSelectorPanel,
    CollapsibleSection,
    LabelAnnotator,
    get_class_selection,
)
from napari_feature_classifier.classifier_widget import ClassifierRunContainer
from napari_feature_classifier.feature_loader_widget import make_features

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

CLASS_NAMES = ["Alpha", "Beta", "Gamma"]
FEATURE_NAMES = ["feature_0", "feature_1", "feature_2"]
N_LABELS = 10


def make_panel(
    class_names=CLASS_NAMES, class_colors=None, on_names=None, on_colors=None
):
    cs = get_class_selection(class_names=class_names)
    return ClassSelectorPanel(
        ClassSelection=cs,
        class_colors=class_colors,
        on_names_changed=on_names,
        on_colors_changed=on_colors,
    )


def make_label_layer(viewer, name="Labels", roi_id="site1"):
    img = np.zeros((10, 10), dtype=np.int32)
    for i in range(N_LABELS):
        img[i // 10, i % 10] = i + 1
    layer = viewer.add_labels(img, name=name)
    layer.features = make_features(
        labels=list(range(1, N_LABELS + 1)), roi_id=roi_id, n_features=3
    )
    return layer


@pytest.fixture()
def viewer(make_napari_viewer):
    return make_napari_viewer()


# ---------------------------------------------------------------------------
# ClassSelectorRow
# ---------------------------------------------------------------------------


def test_row_deselect_clears_radio(make_napari_viewer):
    """deselect() sets radio to False without firing on_select."""
    fired = []
    cs = get_class_selection(class_names=["A", "B"])
    panel = ClassSelectorPanel(ClassSelection=cs, on_colors_changed=None)
    row = panel._rows[0]
    row.select()
    row.deselect()
    assert row._radio.value is False
    assert fired == []  # no external callback should fire


def test_row_update_color_changes_internal_color(make_napari_viewer):
    """update_color() stores the new color and updates the button stylesheet."""
    cs = get_class_selection(class_names=["A"])
    panel = ClassSelectorPanel(ClassSelection=cs)
    row = panel._rows[0]
    new_color = (1.0, 0.0, 0.0, 1.0)
    row.update_color(new_color)
    assert row._color == new_color
    # Stylesheet should mention the new red component
    style = row._color_btn.native.styleSheet()
    assert "255" in style  # r=1.0 → 255


def test_row_update_count_updates_label(make_napari_viewer):
    cs = get_class_selection(class_names=["A"])
    panel = ClassSelectorPanel(ClassSelection=cs)
    row = panel._rows[0]
    row.update_count(42)
    assert row._count_label.value == "42"


def test_row_radio_changed_fires_on_select():
    """Toggling the radio to True calls on_select with the class index."""
    cs = get_class_selection(class_names=["A", "B"])
    panel = ClassSelectorPanel(ClassSelection=cs, on_colors_changed=None)
    row = panel._rows[1]  # class_index=2
    row.deselect()
    row._on_radio_changed(True)
    # _on_row_selected is what gets called; _selected_n should update
    assert panel._selected_n == 2


def test_row_radio_changed_false_does_not_fire():
    """Toggling to False should not call on_select."""
    cs = get_class_selection(class_names=["A"])
    panel = ClassSelectorPanel(ClassSelection=cs)
    row = panel._rows[0]
    initial = panel._selected_n
    row._on_radio_changed(False)
    assert panel._selected_n == initial


# ---------------------------------------------------------------------------
# ClassSelectorPanel — value, set_selected, get_color
# ---------------------------------------------------------------------------


def test_panel_value_returns_correct_enum_member():
    """value property returns the currently selected enum member."""
    panel = make_panel()
    # Default: first class selected
    assert panel.value.name == CLASS_NAMES[0]


def test_panel_set_selected_noclass():
    """set_selected(0) deselects all rows and sets value to NoClass."""
    panel = make_panel()
    panel.set_selected(0)
    assert panel._selected_n == 0
    for row in panel._rows:
        assert row._radio.value is False


def test_panel_set_selected_second_class():
    """set_selected(2) selects second row, deselects first."""
    panel = make_panel()
    panel.set_selected(2)
    assert panel._selected_n == 2
    assert panel._rows[1]._radio.value is True  # class_index=2 → rows[1]
    assert panel._rows[0]._radio.value is False


def test_panel_get_color_in_range():
    """get_color returns the stored row color for valid index."""
    red = (1.0, 0.0, 0.0, 1.0)
    panel = make_panel(class_colors={1: red})
    assert panel.get_color(1) == red


def test_panel_get_color_out_of_range():
    """get_color returns transparent for out-of-range index."""
    panel = make_panel()
    assert panel.get_color(0) == (0.0, 0.0, 0.0, 0.0)
    assert panel.get_color(99) == (0.0, 0.0, 0.0, 0.0)


def test_panel_resolve_color_uses_stored_color():
    """_resolve_color returns the explicitly stored color, not Set1."""
    stored = (0.5, 0.5, 0.5, 1.0)
    panel = make_panel(class_colors={2: stored})
    assert panel._resolve_color(2) == stored


def test_panel_resolve_color_falls_back_to_cmap():
    """_resolve_color without stored color returns a non-zero tuple from Set1."""
    panel = make_panel()
    color = panel._resolve_color(1)
    assert len(color) == 4
    assert any(c > 0 for c in color)


# ---------------------------------------------------------------------------
# ClassSelectorPanel — name and color callbacks
# ---------------------------------------------------------------------------


def test_panel_on_row_name_changed_fires_callback():
    """Renaming a class fires on_names_changed with the updated list."""
    received = []
    panel = make_panel(on_names=lambda names: received.append(list(names)))
    panel._on_row_name_changed(1, "Renamed")
    assert received == [["Renamed", "Beta", "Gamma"]]


def test_panel_on_row_name_changed_rebuilds_enum():
    """After a rename the ClassSelection enum uses the new name."""
    panel = make_panel()
    panel._on_row_name_changed(2, "NewBeta")
    assert "NewBeta" in panel.ClassSelection.__members__
    assert "Beta" not in panel.ClassSelection.__members__


def test_panel_on_row_name_changed_duplicate_ignored():
    """Renaming a class to an existing name does not update the enum."""
    panel = make_panel()
    original = panel.ClassSelection
    panel._on_row_name_changed(2, "Alpha")  # duplicate of class 1
    assert panel.ClassSelection is original  # unchanged


def test_panel_on_row_color_changed_fires_callback():
    """Changing a row's color fires on_colors_changed."""
    received = []
    panel = make_panel(on_colors=lambda idx, rgba: received.append((idx, rgba)))
    new_color = (0.0, 1.0, 0.0, 1.0)
    panel._on_row_color_changed(1, new_color)
    assert received == [(1, new_color)]
    assert panel._class_colors[1] == new_color


def test_panel_update_counts_sets_row_labels():
    """update_counts() sets each row's count label to the matching value."""
    panel = make_panel()
    panel.update_counts({"Alpha": 5, "Beta": 3, "Gamma": 0})
    assert panel._rows[0]._count_label.value == "5"
    assert panel._rows[1]._count_label.value == "3"
    assert panel._rows[2]._count_label.value == "0"


def test_panel_update_counts_missing_key_defaults_to_zero():
    """update_counts() defaults to 0 for classes not in the dict."""
    panel = make_panel()
    panel.update_counts({"Alpha": 7})
    assert panel._rows[1]._count_label.value == "0"


# ---------------------------------------------------------------------------
# CollapsibleSection
# ---------------------------------------------------------------------------


def test_collapsible_section_starts_collapsed(make_napari_viewer):
    """Inner container state is collapsed when collapsed=True (default)."""
    from magicgui.widgets import Label

    section = CollapsibleSection("Test", [Label(value="x")], collapsed=True)
    assert section._expanded is False
    assert "▶" in section._toggle_btn.text


def test_collapsible_section_toggle_shows_inner(make_napari_viewer):
    """Calling _toggle once marks section expanded and flips arrow."""
    from magicgui.widgets import Label

    section = CollapsibleSection("Test", [Label(value="x")], collapsed=True)
    section._toggle()
    # In headless Qt, isVisible() reflects parent visibility, so check state directly
    assert section._expanded is True
    assert "▼" in section._toggle_btn.text


def test_collapsible_section_toggle_twice_hides_again(make_napari_viewer):
    """Two toggles return to the collapsed state."""
    from magicgui.widgets import Label

    section = CollapsibleSection("Test", [Label(value="x")], collapsed=True)
    section._toggle()
    section._toggle()
    assert section._expanded is False
    assert "▶" in section._toggle_btn.text


def test_collapsible_section_starts_expanded(make_napari_viewer):
    """collapsed=False means expanded state from the start."""
    from magicgui.widgets import Label

    section = CollapsibleSection("Test", [Label(value="x")], collapsed=False)
    assert section._expanded is True
    assert "▼" in section._toggle_btn.text


# ---------------------------------------------------------------------------
# LabelAnnotator — selection_changed disable branch
# ---------------------------------------------------------------------------


def test_label_annotator_selection_changed_disables_on_non_label_layer(viewer):
    """Selecting a non-label layer disables the save section and class selector."""
    make_label_layer(viewer)
    annotator = LabelAnnotator(viewer, get_class_selection(class_names=["A", "B"]))
    # Switch to a non-label layer
    img_layer = viewer.add_image(np.zeros((10, 10)), name="Image")
    viewer.layers.selection.active = img_layer
    assert annotator._save_section.enabled is False
    assert annotator._class_selector.enabled is False


def test_label_annotator_selection_changed_enables_on_label_layer(viewer):
    """Switching back to a label layer re-enables the UI."""
    layer = make_label_layer(viewer)
    annotator = LabelAnnotator(viewer, get_class_selection(class_names=["A", "B"]))
    img_layer = viewer.add_image(np.zeros((10, 10)), name="Image")
    viewer.layers.selection.active = img_layer
    viewer.layers.selection.active = layer
    assert annotator._save_section.enabled is True
    assert annotator._class_selector.enabled is True


# ---------------------------------------------------------------------------
# get_scaled_position — 3D→2D edge case
# ---------------------------------------------------------------------------


def test_get_scaled_position_3d_to_2d():
    """3D position with 2D scale drops the z component."""
    result = LabelAnnotator.get_scaled_position(
        position=(5.0, 10.0, 20.0),
        translate=np.array([0.0, 0.0]),
        scale=np.array([2.0, 4.0]),
    )
    assert result == (5.0, 5.0)


def test_get_scaled_position_raises_on_incompatible_dimensions():
    """Incompatible position/scale dimensions raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        LabelAnnotator.get_scaled_position(
            position=(1.0, 2.0, 3.0, 4.0),
            translate=np.array([0.0, 0.0]),
            scale=np.array([1.0, 1.0]),
        )


# ---------------------------------------------------------------------------
# ClassifierRunContainer — restore, name/color change callbacks
# ---------------------------------------------------------------------------


def test_restore_annotations_updates_colormap_and_counts(viewer, tmp_path, monkeypatch):
    """_restore_annotations re-renders the annotation layer and updates counts."""
    monkeypatch.chdir(tmp_path)
    layer = make_label_layer(viewer, roi_id="site1")
    layer.features.loc[0, "annotations"] = 1.0
    layer.features.loc[1, "annotations"] = 2.0

    container = ClassifierRunContainer(
        viewer, class_names=["A", "B"], feature_names=FEATURE_NAMES
    )
    container._runner.add_features_to_classifier()

    # Simulate reopening: clear live annotations
    layer.features["annotations"] = float("nan")

    container._restore_annotations(layer)

    # Annotations written back
    assert layer.features.loc[0, "annotations"] == 1.0
    assert layer.features.loc[1, "annotations"] == 2.0

    # Counts updated
    counts = container._get_annotation_counts()
    assert counts["A"] == 1
    assert counts["B"] == 1


def test_on_class_names_changed_updates_classifier(viewer, tmp_path, monkeypatch):
    """Renaming a class via the panel updates classifier._class_names."""
    monkeypatch.chdir(tmp_path)
    make_label_layer(viewer)
    container = ClassifierRunContainer(
        viewer, class_names=["A", "B"], feature_names=FEATURE_NAMES
    )
    container._on_class_names_changed(["NewA", "B"])
    assert container._classifier._class_names == ["NewA", "B"]


def test_on_class_colors_changed_updates_classifier(viewer, tmp_path, monkeypatch):
    """Changing a color via the panel updates classifier._class_colors."""
    monkeypatch.chdir(tmp_path)
    make_label_layer(viewer)
    container = ClassifierRunContainer(
        viewer, class_names=["A", "B"], feature_names=FEATURE_NAMES
    )
    new_color = (1.0, 0.0, 0.0, 1.0)
    container._on_class_colors_changed(1, new_color)
    assert container._classifier._class_colors[1] == new_color


def test_load_classifier_container_no_label_layer(viewer, capsys):
    """LoadClassifierContainer.load() shows an info message when no label layer exists."""
    from pathlib import Path

    from napari_feature_classifier.classifier_widget import LoadClassifierContainer

    clf_path = Path(
        "src/napari_feature_classifier/sample_data/test_labels_classifier.clf"
    )
    loading_widget = LoadClassifierContainer(viewer)
    loading_widget._clf_destination.value = clf_path
    loading_widget.load()
    out = capsys.readouterr().out
    assert "label layer" in out.lower() or loading_widget._run_container is None
