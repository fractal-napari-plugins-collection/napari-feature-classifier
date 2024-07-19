"""Utils function for the classifier"""
from functools import lru_cache
import logging
import math
from pathlib import Path

# import warnings
import pandas as pd
from napari.utils.notifications import show_info
from matplotlib.colors import ListedColormap
import matplotlib
import numpy as np
import napari
from qtpy.QtWidgets import QMessageBox  # pylint: disable=E0611

# from napari._qt.dialogs.qt_notification import NapariQtNotification
# from napari._qt.qt_event_loop import _ipython_has_eventloop


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
        from IPython import get_ipython  # pylint: disable-msg=C0415

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
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
    new_colors = np.array(matplotlib.colormaps[matplotlib_colormap].colors).astype(
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
    label_layer, feature_col, display_layer, label_column, cmap
):
    """
    Reset the colormap based on the annotations in
    label_layer.features['annotation'] and sends the updated colormap
    to the annotation label layer
    """
    colors = cmap(label_layer.features[feature_col].astype(float) / len(cmap.colors))
    colordict = dict(zip(label_layer.features[label_column], colors))
    display_layer.color = colordict
    display_layer.opacity = 1.0
    display_layer.color_mode = "direct"


#     # Check if it runs in napari
#     # This currently triggers an exception.
#     # Find a new way to ensure the warning is also shown in the napari
#     # interface    # if _ipython_has_eventloop():
#     NapariQtNotification(message, 'INFO').show()


# def napari_warn(message):
#     # Wrapper function to ensure a message o
#     warnings.warn(message)
#     show_info(message)
#     print('test')
#     # This currently triggers an exception.
#     # Find a new way to ensure the warning is also shown in the napari
#     # interface
#     if _ipython_has_eventloop():
#         pass
#         # NapariQtNotification(message, 'WARNING').show()
#
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


def get_valid_label_layers(viewer) -> list[str]:
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
            class_names.append(np.NaN)
        else:
            class_names.append(ClassSelection(annotation).name)
    df["annotation_names"] = class_names
    return df
