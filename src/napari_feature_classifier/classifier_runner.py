"""ML orchestration and prediction layer management for the classifier plugin.

These classes have no Qt dependency and can be unit-tested with a headless
napari viewer.
"""

from collections.abc import Callable

import napari
import napari.layers
import napari.viewer
import numpy as np
import pandas as pd

from napari_feature_classifier.classifier import Classifier


class ClassifierRunner:
    """
    Handles all ML orchestration: feature collection from viewer layers,
    training pipeline coordination, and prediction merging.

    Reads from and writes to layer.features. Has zero UI dependency.

    Parameters
    ----------
    viewer: napari.viewer.Viewer
        The current napari Viewer instance.
    classifier: Classifier
        The classifier instance to train and run predictions with.
    label_column: str
        Column name for object labels in layer.features. Default "label".
    roi_id_column: str
        Column name for ROI identifiers in layer.features. Default "roi_id".
    """

    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        classifier: Classifier,
        label_column: str = "label",
        roi_id_column: str = "roi_id",
    ) -> None:
        self._viewer = viewer
        self._classifier = classifier
        self._label_column = label_column
        self._roi_id_column = roi_id_column

    @property
    def feature_names(self) -> list[str]:
        return self._classifier.get_feature_names()

    def add_features_to_classifier(self) -> None:
        """
        Collect annotated features from all label layers and pass to the
        classifier.

        Iterates all label layers that have an 'annotations' column. If a
        layer has a 'roi_id' column the value is used as key; otherwise the
        layer name is used as a fallback.
        """
        dict_of_features: dict[str, pd.DataFrame] = {}
        for layer in self._viewer.layers:
            if (
                isinstance(layer, napari.layers.Labels)
                and len(layer.features) > 0
                and "annotations" in layer.features.columns
            ):
                # TODO: Add extra checks that it contains valid features?
                if "roi_id" in layer.features.columns:
                    roi_ids = layer.features["roi_id"].unique()
                    if len(roi_ids) > 1:
                        raise NotImplementedError(
                            f"{layer=} contained no-unique roi_ids: {roi_ids}"
                        )
                    roi_id = roi_ids[0]
                    dict_of_features[roi_id] = layer.features
                else:
                    # TODO: Consider label-layer hashing here instead of
                    # using the layer name as roi_id
                    dict_of_features[layer.name] = layer.features
        self._classifier.add_dict_of_features(dict_of_features)

    def make_predictions(self) -> None:
        """
        Run predictions on all relevant label layers and merge the results
        back into each layer's features as a 'prediction' column.

        Does NOT update the colormap — the coordinator is responsible for
        calling PredictionLayerManager.setup() after this method.
        """
        relevant_label_layers = self.get_relevant_label_layers()

        prediction_dfs: dict[str, pd.DataFrame] = {}
        for label_layer in relevant_label_layers:
            roi_id = self.get_layer_roi_id(label_layer)
            if roi_id in prediction_dfs:
                raise ValueError(
                    f"Duplicate roi_id {roi_id} found in {label_layer.name}. "
                    "It's already present as the roi_id of another label layer."
                )
            prediction_dfs[roi_id] = self.get_relevant_features(
                label_layer.features, set_index=False
            )

        prediction_results_dict = self._classifier.predict_on_dict(prediction_dfs)

        for label_layer in relevant_label_layers:
            roi_id = self.get_layer_roi_id(label_layer)
            if "prediction" in label_layer.features.columns:
                label_layer.features.drop(columns=["prediction"], inplace=True)
            label_layer.features = label_layer.features.merge(
                prediction_results_dict[roi_id],
                left_on=[self._roi_id_column, self._label_column],
                right_index=True,
                how="outer",
            )

    def get_relevant_label_layers(self) -> list[napari.layers.Labels]:
        """
        Return label layers that have both 'label' and 'roi_id' columns,
        excluding the "Annotations" and "Predictions" layers.
        """
        required_columns = [self._label_column, self._roi_id_column]
        excluded_names = ["Annotations", "Predictions"]
        return [
            layer
            for layer in self._viewer.layers
            if (
                isinstance(layer, napari.layers.Labels)
                and layer.name not in excluded_names
                and layer.features is not None
                and all(c in layer.features.columns for c in required_columns)
            )
        ]

    def get_layer_roi_id(self, label_layer: napari.layers.Labels) -> str:
        """
        Extract the single roi_id value from a layer's features.

        Raises NotImplementedError if the layer contains more than one
        unique roi_id.
        """
        roi_ids = label_layer.features[self._roi_id_column].unique()
        if len(roi_ids) > 1:
            raise NotImplementedError(
                f"{label_layer=} contained no-unique roi_ids: {roi_ids}"
            )
        return roi_ids[0]

    def get_relevant_features(
        self,
        df: pd.DataFrame,
        filter_annotations: bool = False,
        set_index: bool = False,
    ) -> pd.DataFrame:
        """
        Slice a features DataFrame to only the columns needed by the
        classifier.

        Parameters
        ----------
        df:
            Full features DataFrame from a label layer.
        filter_annotations:
            If True, only return rows that have a non-NaN annotation value.
        set_index:
            If True, set [roi_id_column, label_column] as a MultiIndex.
        """
        if not filter_annotations:
            df_relevant = df[
                [*self.feature_names, self._label_column, self._roi_id_column]
            ]
        else:
            df_relevant = df.loc[
                df["annotations"].notna(),
                [
                    *self.feature_names,
                    self._label_column,
                    self._roi_id_column,
                    "annotations",
                ],
            ]
        if set_index:
            df_relevant = df_relevant.set_index(
                [self._roi_id_column, self._label_column]
            )
        return df_relevant


class PredictionLayerManager:
    """
    Owns the lifecycle of the "Predictions" napari layer: creation, geometry
    synchronisation, colormap updates, and layer reordering.

    Has no Qt dependency.

    Parameters
    ----------
    viewer: napari.viewer.Viewer
        The current napari Viewer instance.
    label_column: str
        Column name for object labels in layer.features. Default "label".
    """

    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        label_column: str = "label",
    ) -> None:
        self._viewer = viewer
        self._label_column = label_column
        self._prediction_layer: napari.layers.Labels | None = None
        # Optional per-class color resolver; None = use Set1 colormap fallback.
        # Set by ClassifierRunContainer after constructing the annotator.
        self._color_resolver: Callable[[int], tuple] | None = None

        # Remove any stale Predictions layer left from a previous session
        for layer in list(self._viewer.layers):
            if isinstance(layer, napari.layers.Labels) and layer.name == "Predictions":
                self._viewer.layers.remove(layer)

    @property
    def prediction_layer(self) -> napari.layers.Labels:
        """The managed Predictions layer. Raises if setup() has not been called."""
        if self._prediction_layer is None:
            raise RuntimeError(
                "PredictionLayerManager.setup() must be called before "
                "accessing prediction_layer."
            )
        return self._prediction_layer

    def set_visible(self, visible: bool) -> None:
        """Show or hide the Predictions layer."""
        self.prediction_layer.visible = visible

    def setup(self, label_layer: napari.layers.Labels) -> None:
        """
        Full setup: create the Predictions layer if absent, sync geometry and
        colormap to label_layer, then reorder layers.

        Call this at initialisation time and after make_predictions().
        """
        if "Predictions" not in [x.name for x in self._viewer.layers]:
            self._add_prediction_layer(label_layer)
        self._ensure_prediction_column(label_layer)
        self._sync_geometry(label_layer)
        self._update_colormap(label_layer)
        # Reorder — occasionally fails with napari internal errors; safe to ignore
        try:
            self._reorder_layers(label_layer)
        except:  # noqa
            pass

    def sync(self, label_layer: napari.layers.Labels) -> None:
        """
        Sync geometry and colormap to label_layer without creating the layer
        or reordering.

        Call this from selection_changed() when switching to a different layer.
        """
        if self._prediction_layer is None:
            return
        self._ensure_prediction_column(label_layer)
        self._sync_geometry(label_layer)
        self._update_colormap(label_layer)

    # --- private helpers ---

    def _add_prediction_layer(self, label_layer: napari.layers.Labels) -> None:
        self._prediction_layer = self._viewer.add_labels(
            label_layer.data,
            scale=label_layer.scale,
            name="Predictions",
            translate=label_layer.translate,
        )
        self._prediction_layer.contour = 2

    def _ensure_prediction_column(self, label_layer: napari.layers.Labels) -> None:
        """Add a NaN-filled 'prediction' column to layer.features if absent."""
        if "prediction" not in label_layer.features:
            unique_labels = np.unique(label_layer.data)[1:]
            predict_df = pd.DataFrame(
                {self._label_column: unique_labels, "prediction": np.NaN}
            )
            if self._label_column in label_layer.features.columns:
                label_layer.features = label_layer.features.merge(
                    predict_df, on=self._label_column, how="outer"
                )
            else:
                label_layer.features = pd.concat(
                    [label_layer.features, predict_df], axis=1
                )

    def _sync_geometry(self, label_layer: napari.layers.Labels) -> None:
        """Sync the prediction layer's data, scale, and translate to label_layer."""
        self.prediction_layer.data = label_layer.data
        self.prediction_layer.scale = label_layer.scale
        self.prediction_layer.translate = label_layer.translate

    def _update_colormap(self, label_layer: napari.layers.Labels) -> None:
        from napari_feature_classifier.utils import (
            get_colormap,
            reset_display_colormaps,
        )

        if self._color_resolver is not None:
            reset_display_colormaps(
                label_layer,
                feature_col="prediction",
                display_layer=self.prediction_layer,
                label_column=self._label_column,
                color_resolver=self._color_resolver,
            )
        else:
            reset_display_colormaps(
                label_layer,
                feature_col="prediction",
                display_layer=self.prediction_layer,
                label_column=self._label_column,
                cmap=get_colormap(),
            )

    def _reorder_layers(self, reference_layer: napari.layers.Labels) -> None:
        """Ensure Predictions, Annotations, and reference layer are ordered correctly."""
        all_layers = list(self._viewer.layers)
        indices_to_move = []
        if "Predictions" in self._viewer.layers:
            indices_to_move.append(self._viewer.layers.index("Predictions"))
        if "Annotations" in self._viewer.layers:
            indices_to_move.append(self._viewer.layers.index("Annotations"))
        if reference_layer.name in self._viewer.layers:
            indices_to_move.append(self._viewer.layers.index(reference_layer.name))

        remaining_indices = [
            i for i in range(len(all_layers)) if i not in indices_to_move
        ]
        remaining_indices.reverse()
        new_order = indices_to_move + remaining_indices
        new_order.reverse()
        self._viewer.layers.move_multiple(new_order)
