# %%
from os import PathLike
from typing import Callable, Sequence, TypeAlias

import napari
import numpy as np
import pandas as pd
import pandera as pa
from magicgui.widgets import Container, FileEdit, PushButton
from napari.utils.notifications import show_info
from pandera.typing import DataFrame, Index


class LabelFeatureSchema(pa.SchemaModel):
    label: Index[int] = pa.Field(coerce=True, unique=True, check_name=False)


@pa.check_types
def load_features(fn: PathLike) -> DataFrame[LabelFeatureSchema]:
    # return pd.DataFrame(data=np.random.rand(5, 5), columns=list("abcde"), index=list(range(1, 10, 2)))
    return DataFrame[LabelFeatureSchema](
        data=np.random.rand(5, 5), columns=list("abcde"), index=list(range(5))
    )


@pa.check_types
def make_features(
    labels: Sequence[int], n_features: int = 10, seed: int = 42
) -> DataFrame[LabelFeatureSchema]:
    columns = [f"feature_{i}" for i in range(n_features)]
    rng = np.random.default_rng(seed=seed)
    features = rng.random(size=(len(labels), n_features))
    return DataFrame[LabelFeatureSchema](index=labels, columns=columns, data=features)


FeatureLoaderFn: TypeAlias = Callable[[PathLike[str]], DataFrame[LabelFeatureSchema]]


class LoadFeaturesContainer(Container):
    def __init__(self, viewer: napari.viewer.Viewer):
        self._viewer = viewer
        self._load_destination = FileEdit(value="sample_data/test_df.csv", mode="r")
        self._load_button = PushButton(label="Load Features")
        super().__init__(widgets=[self._load_destination, self._load_button])
        self._load_button.clicked.connect(self.load)

    def load(self):
        fn = self._load_destination.value
        df = pd.read_csv(fn).set_index("label")
        if self._label_layer.features.index.empty:
            self._label_layer.features.index = np.unique(self._label_layer.data)[1:]
        self._label_layer.features = self._label_layer.features.join(df, how="left")
        show_info("loading csv...")
        print(self._label_layer.features)


# %%
