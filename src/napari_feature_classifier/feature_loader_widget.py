"""Widget for loading features from csv file."""

import warnings
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
import pandera.pandas as pa
from magicgui import magic_factory
from napari.layers import Labels
from napari.types import LayerDataTuple
from pandera.typing import DataFrame, Series

from napari_feature_classifier.utils import napari_info


# pandera.SchemaModel is now deprecated
# see https://github.com/unionai-oss/pandera/releases/tag/v0.20.0
class LabelFeatureSchema(pa.DataFrameModel):
    # roi_id: Series[str] = pa.Field(coerce=True, unique=False)
    label: Series[int] = pa.Field(coerce=True, unique=True)


@pa.check_types
def load_features_csv(
    fn: PathLike, index_column_or_columns: str | list[str] = "label"
) -> DataFrame[LabelFeatureSchema]:
    df = pd.read_csv(fn)
    if isinstance(index_column_or_columns, str):
        assert (
            index_column_or_columns in df
        ), f"missing index column `{index_column_or_columns}` in csv file."
        return DataFrame[LabelFeatureSchema](
            df.rename(columns={index_column_or_columns: "label"})
        )
    # return pd.DataFrame(data=np.random.rand(5, 5), columns=list("abcde"),
    # index=list(range(1, 10, 2)))
    return DataFrame[LabelFeatureSchema](df)


@pa.check_types
def make_features(
    labels: Sequence[int] | np.ndarray,
    roi_id: str = "id",
    n_features: int = 10,
    seed: int = 42,
) -> DataFrame[LabelFeatureSchema]:
    columns = [f"feature_{i}" for i in range(n_features)]
    rng = np.random.default_rng(seed=seed)
    features = rng.random(size=(len(labels), n_features))
    data = {
        **{"roi_id": roi_id, "label": labels},
        **{
            column: feature
            for column, feature in zip(columns, features.T, strict=False)
        },
    }
    return DataFrame[LabelFeatureSchema](data)


FeatureLoaderFn: TypeAlias = Callable[[PathLike[str]], DataFrame[LabelFeatureSchema]]


@magic_factory(call_button="load features")
def load_features_factory(
    layer: Labels, path: Path, loader: FeatureLoaderFn = load_features_csv
) -> LayerDataTuple:
    df = loader(path)  # pylint: disable=C0103
    image_labels = np.unique(np.asarray(layer.data))[1:]
    feature_labels = df["label"].values
    if len(set(image_labels).symmetric_difference(feature_labels)) != 0:
        warn_str = "Label image labels do not match with feature table.\n"
        "Label objects with no features: "
        f"{sorted(set(image_labels).difference(feature_labels))}\n"
        "Features with no label objects: "
        f"{sorted(set(feature_labels).difference(image_labels))}"
        napari_info(warn_str)
        warnings.warn(warn_str, stacklevel=2)
    napari_info(f'Loaded features and attached them to "{layer}" layer')
    return (layer.data, {"name": layer.name, "features": df}, "labels")  # type: ignore[return-value]
