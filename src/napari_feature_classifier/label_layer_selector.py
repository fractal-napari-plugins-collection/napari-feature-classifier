from magicgui.widgets import Container, ComboBox, create_widget
import napari

from typing import Optional, Sequence, cast

class LabelLayerSelector(Container):
    def __init__(self, viewer: napari.viewer.Viewer):
        self._viewer = viewer
        self._lbl_combo = ComboBox(
            label="Current Label Layer",
            choices=self._get_appropriate_label_layer_names()
        )

        super().__init__(widgets=[self._lbl_combo])

        self._viewer.layers.events.inserted.connect(self.update_layer_choices)
        self._viewer.layers.events.removed.connect(self.update_layer_choices)
        # FIXME: Connect to layer renaming events to update the layer list 
        # (not trivial, see https://forum.image.sc/t/event-handling-in-napari/46539/4)

        # Update selections when the active layer changes
        self._viewer.layers.selection.events.changed.connect(self._active_changed)
        # Update label layer selection when the user uses the ComboBox
        self._lbl_combo.changed.connect(self._update_layer_selection)

        self._active_changed()

    def _get_appropriate_label_layer_names(self) -> list[str]:
        # Select label layers that are not `Annotations` or `Predictions`.
        return [
            layer.name
            for layer in self._viewer.layers
            if isinstance(layer, napari.layers.Labels)
            and layer.name not in ["Annotations", "Predictions"]
        ]
    
    def update_layer_choices(self, event):
        self._lbl_combo._default_choices = self._get_appropriate_label_layer_names()

    def _active_changed(self):
        current_layer_proxy = self._viewer.layers.selection.active
        print(f'Run active changed for {current_layer_proxy}')
        print(f'The current label layer selection was :{self.get_selected_label_layer()}')        
        if current_layer_proxy is None:
            return
        elif (
            current_layer_proxy.__class__ == napari.layers.Labels
            and current_layer_proxy.name != "Annotations"
            and current_layer_proxy.name != "Predictions"
        ):
            self._lbl_combo.value = current_layer_proxy.name
        elif(current_layer_proxy.name == "Annotations" 
             or current_layer_proxy.name == "Predictions"):
            # FIXME: Adding this leads to weird behavior of not having any 
            # layers selected upon Annotator init
            # Don't allow the user to select the Annotations or 
            # Predicitions layer
            # self._viewer.layers.selection.active = None
            # self._viewer.layers.selection.add(self.get_selected_label_layer())
            # self._viewer.layers.selection.clear()
            # self._viewer.layers.selection.add(self.get_selected_label_layer())
            pass
        else:
            return
    
    def _update_layer_selection(self, event):
        print(f'Update label layer to :{self.get_selected_label_layer()}')
        self._viewer.layers.selection.active = self.get_selected_label_layer()
        # self._viewer.layers.selection.clear()
        # self._viewer.layers.selection.add(self.get_selected_label_layer())

    def get_selected_label_layer(self) -> Optional[napari.layers.Labels]:
        if self._lbl_combo.value is None:
            return None
        else:
            return self._viewer.layers[self._lbl_combo.value]