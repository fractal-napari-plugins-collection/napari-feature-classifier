"""Utils function for the classifier"""

import logging
import math
from functools import lru_cache
from pathlib import Path

import matplotlib
import napari
import napari.layers
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QMessageBox  # pylint: disable=E0611


@lru_cache(maxsize=16)
def get_df(path):
    """
    Pandas csv reader function with caching

    Parameters
    ----------
    path: str or Path
        Path to the csv file to be loaded
    """
    return pd.read_csv(path)


def in_notebook():
    """
    Checks whether the plugin is run from within a jupyter notebook

    Returns
    -------
    boolean
        True if it's running in a jupyter notebook
    """

    # Check if I'm running in jupyter notebook, from here:
    # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    try:
        from IPython import (
            get_ipython,  # type: ignore[attr-defined]  # pylint: disable-msg=C0415
        )

        if "IPKernelApp" not in get_ipython().config:  # type: ignore[union-attr]  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def get_colormap(matplotlib_colormap="Set1"):
    """
    Generates colormaps depending on the number of classes
    """
    new_colors = np.array(matplotlib.colormaps[matplotlib_colormap].colors).astype(  # type: ignore[attr-defined]
        np.float32
    )
    cmap_np = np.zeros(
        shape=(new_colors.shape[0] + 1, new_colors.shape[1] + 1), dtype=np.float32
    )
    cmap_np[1:, :-1] = new_colors
    cmap_np[1:, -1] = 1
    cmap = ListedColormap(cmap_np)
    return cmap


def reset_display_colormaps(
    label_layer,
    feature_col,
    display_layer,
    label_column,
    cmap=None,
    color_resolver=None,
):
    """
    Reset the colormap based on the annotations in
    label_layer.features[feature_col] and apply it to display_layer.

    Either `cmap` or `color_resolver` must be provided.

    Parameters
    ----------
    cmap : matplotlib colormap, optional
        Used when color_resolver is None. Color = cmap(value / len(cmap.colors)).
    color_resolver : Callable[[int], RGBA], optional
        If provided, called with the integer annotation/prediction value to
        return an RGBA tuple. Takes precedence over cmap.
    """
    feature_values = label_layer.features[feature_col]
    if color_resolver is not None:
        colors = [
            color_resolver(int(v))
            if not (isinstance(v, float) and math.isnan(v))
            else (0.0, 0.0, 0.0, 0.0)
            for v in feature_values
        ]
    else:
        colors = cmap(feature_values.astype(float) / len(cmap.colors))  # type: ignore[arg-type,call-arg]
    colordict = dict(zip(label_layer.features[label_column], colors, strict=False))
    colordict[None] = [0, 0, 0, 0]
    display_layer.colormap = DirectLabelColormap(color_dict=colordict)
    display_layer.opacity = 1.0


def napari_info(message):
    """
    Info message wrapper.
    Ensures info is shown in napari (when napari is run from the command line)
    or printed (when napari is run from a jupyter notebook)
    If napari show_info can't be called (e.g. napari isn't running),
    it's skipped

    message
    ----------
    path: str
        Message to be shown to the user
    """
    try:
        show_info(message)
    except:  # pylint: disable=bare-except # noqa #E722
        print(message)
    # TODO: Would be better to check if it's running in napari and print in all
    # other cases (e.g. if someone runs the classifier form a script).
    # But can't make that work at the moment
    if in_notebook():
        print(message)


class NapariHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        napari_info(log_entry)


def get_valid_label_layers(viewer) -> list[napari.layers.Labels]:
    """
    Get a list of label layers that are not `Annotations` or `Predictions`.
    """
    return [
        layer
        for layer in viewer.layers
        if isinstance(layer, napari.layers.Labels)
        and layer.name not in ["Annotations", "Predictions"]
    ]


def get_selected_or_valid_label_layer(viewer) -> napari.layers.Labels:
    """
    Get the selected label layer, or the first valid label layer.
    This is None if no layer or multiple layers are selected.
    """
    selected_layer = viewer.layers.selection.active
    valid_layers = get_valid_label_layers(viewer=viewer)
    if selected_layer and selected_layer in valid_layers:
        return viewer.layers[selected_layer.name]
    if len(valid_layers) > 0:
        return valid_layers[0]
    raise NotImplementedError("No valid label layers were found")


def overwrite_check_passed(file_path, output_type: str = ""):
    """
    If a file already exists, ask whether it should be overwritten.
    """
    if Path(file_path).exists():
        msg_box = QMessageBox()
        msg_box.setText(
            "Do you you want to overwrite the "
            f"existing {output_type}: "
            f"{file_path}?"
        )
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        msg_box.setDefaultButton(QMessageBox.Yes)

        response = msg_box.exec_()
        if not response == QMessageBox.Yes:
            return False
    return True


# pylint: disable=C0103
def add_annotation_names(df, ClassSelection):
    """
    Add a column with the actual annotation names to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with annotations column.
    ClassSelection : Enum
        Enum with the class names.

    Returns
    -------
    pd.DataFrame
    """
    class_names = []
    for annotation in df["annotations"]:
        if math.isnan(annotation):
            class_names.append(float("nan"))
        else:
            class_names.append(ClassSelection(annotation).name)
    df["annotation_names"] = class_names
    return df
