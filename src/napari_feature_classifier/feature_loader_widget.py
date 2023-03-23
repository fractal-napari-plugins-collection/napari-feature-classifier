# %%
from os import PathLike
from typing import Callable, Sequence, TypeAlias

import napari
import numpy as np
import pandas as pd
import pandera as pa
from magicgui.widgets import Container, FileEdit, PushButton
from napari.utils.notifications import show_info
from pandera.typing import DataFrame, Series


class LabelFeatureSchema(pa.SchemaModel):
    roi_id: Series[str] = pa.Field(coerce=True, unique=False)
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
        return DataFrame[LabelFeaturesSchema](
            df.rename(columns={index_column_or_columns: "label"})
        )
    # return pd.DataFrame(data=np.random.rand(5, 5), columns=list("abcde"), index=list(range(1, 10, 2)))
    return DataFrame[LabelFeatureSchema](df)


@pa.check_types
def make_features(
    labels: Sequence[int], roi_id: str = "id", n_features: int = 10, seed: int = 42
) -> DataFrame[LabelFeatureSchema]:
    columns = [f"feature_{i}" for i in range(n_features)]
    rng = np.random.default_rng(seed=seed)
    features = rng.random(size=(len(labels), n_features))
    data = {
        **{"roi_id": roi_id, "label": labels},
        **{column: feature for column, feature in zip(columns, features.T)},
    }
    return DataFrame[LabelFeatureSchema](data)


FeatureLoaderFn: TypeAlias = Callable[[PathLike[str]], DataFrame[LabelFeatureSchema]]


class LoadFeaturesContainer(Container):
    def __init__(
        self,
        labels_layer: napari.viewer.Viewer,
        loader: FeatureLoaderFn = load_features_csv,
    ):
        self._labels_layer = labels_layer
        self._loader = loader
        self._load_destination = FileEdit(value="sample_data/test_df.csv", mode="r")
        self._load_button = PushButton(label="Load Features")
        super().__init__(widgets=[self._load_destination, self._load_button])
        self._load_button.clicked.connect(self.load)

    def load(self):
        show_info("loading csv...")
        fn = self._load_destination.value
        df = self._loader(fn)
        if self._labels_layer.features.index.empty:
            self._labels_layer.features.index = np.unique(self._label_layer.data)[1:]
        self._label_layer.features = self._label_layer.features.join(df, how="left")
        print(self._label_layer.features)
