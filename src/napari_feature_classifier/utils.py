"""Utils function for the classifier"""
from functools import lru_cache

# import warnings
import pandas as pd
from napari.utils.notifications import show_info

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


# def napari_warn(message):
#     # Wrapper function to ensure a message o
#     warnings.warn(message)
#     show_info(message)
#     print('test')
#     # TODO: This currently triggers an exception.
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
    except: # pylint: disable=bare-except
        pass
    print(message)


#     # TODO: This currently triggers an exception.
#     # Find a new way to ensure the warning is also shown in the napari
#     # interface    # if _ipython_has_eventloop():
#     NapariQtNotification(message, 'INFO').show()
