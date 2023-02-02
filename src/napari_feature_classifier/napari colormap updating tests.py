# napari colormap updating tests
import numpy as np
import napari


color_dict = {1: np.array([0., 1., 0., 1.]), 2: np.array([0., 1., 0., 1.]), 3: np.array([0., 0., 0., 0.]), 4: np.array([1., 0., 0., 1.]), 5: np.array([0., 1., 0., 1.]), 6: np.array([0., 0., 0., 0.]), 7: np.array([0., 0., 0., 0.]), 8: np.array([0., 0., 0., 0.]), 9: np.array([0., 0., 0., 0.]), 10: np.array([0., 0., 0., 0.]), 11: np.array([0., 0., 0., 0.]), 12: np.array([0., 0., 0., 0.]), 13: np.array([0., 0., 0., 0.]), 14: np.array([0., 0., 0., 0.]), 15: np.array([0., 0., 0., 0.]), 16: np.array([0., 0., 0., 0.])}
lbls = imageio.v2.imread('sample_data/test_labels.tif')
viewer = napari.Viewer()
label_layer = viewer.add_labels(lbls)
label_layer.color = color_dict
label_layer.color_mode = 'direct'

