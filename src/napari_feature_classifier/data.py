from hashlib import blake2b
from pathlib import Path
from typing import List

import pandas as pd


def test_set_check(identifier: str, test_ratio: float) -> bool:
    digest64 = int(blake2b(identifier.encode('ascii'), digest_size=8).hexdigest(), base=16)
    return digest64 > (test_ratio * 2 ** 64)


def is_part_of_train_set(index: pd.MultiIndex,
                         test_ratio: float = 0.2) -> pd.Series:  # pylint: disable-msg=C0103, C0116
    concat_index = index.map(lambda x: '_'.join(map(str, x)))
    return pd.Series(index=index,
                     data=[test_set_check(identifier, test_ratio=test_ratio) for identifier in concat_index])


class ClassifierData:
    def __init__(self, features: pd.DataFrame, index_columns: List[str], training_feature_names: List[str]):
        self.index_columns = index_columns
        self.features = self.validate_index(features)
        self.training_feature_names = training_feature_names

        self._class_annotations = pd.Series(index=self.features.index, name='annotations', dtype=pd.Int64Dtype())
        self._class_predictions = pd.Series(index=self.features.index, name='annotations', dtype=pd.Int64Dtype())
        self._is_training_data = is_part_of_train_set(self.features.index)

    @classmethod
    def from_path(cls, path: Path, index_columns: List[str], training_feature_names: List[str]):
        features = pd.read_csv(path)
        return cls(features, index_columns, training_feature_names)

    def append_data_frame(self, new_data: pd.DataFrame) -> None:
        assert set(self.training_feature_names).issubset(new_data.columns)
        self.features = pd.concat([self.features, self.validate_index(new_data)]).drop_duplicates()
        current_index = list(self._class_annotations.index)
        missing_index = list(self.features.index.difference(self._class_annotations.index))
        self._class_annotations = self._class_annotations.reindex(current_index + missing_index)
        self._class_predictions = self._class_predictions.reindex(current_index + missing_index)

    def validate_index(self, features: pd.DataFrame) -> pd.DataFrame:
        if list(features.index.names) == [None]:
            return features.set_index(self.index_columns)
        return features.reset_index().set_index(self.index_columns)

    def load_data_frame(self, path: Path) -> None:
        new_data = pd.read_csv(path)
        self.append_data_frame(new_data)

    @property
    def is_training_data(self) -> pd.Series:
        self._is_training_data = is_part_of_train_set(self.features.index)
        return self._is_training_data

    @property
    def training_data(self) -> pd.DataFrame:
        return self.features.loc[self.is_training_data & ~self._class_annotations.isna(), self.training_feature_names]

    @property
    def test_data(self) -> pd.DataFrame:
        return self.features.loc[~self.is_training_data & ~self._class_annotations.isna(), self.training_feature_names]

    @property
    def training_annotations(self) -> pd.Series:
        return self._class_annotations.loc[self.is_training_data & ~self._class_annotations.isna()]

    @property
    def test_annotations(self) -> pd.Series:
        return self._class_annotations.loc[~self.is_training_data & ~self._class_annotations.isna()]

    @property
    def annotations(self) -> pd.Series:
        return self._class_annotations

    @property
    def predictions(self) -> pd.Series:
        return self._class_predictions

    def update_annotations(self, annotations: pd.Series, site_id: str):
        for label, annotation in annotations.dropna().items():
            self.annotations.loc[(site_id, label)] = annotation
        pass
