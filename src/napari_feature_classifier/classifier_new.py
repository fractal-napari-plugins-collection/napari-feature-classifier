# %%
# FIXME: Get rid of show_info in classifier class
import random
import string
from itertools import chain
from typing import Sequence
from zlib import crc32

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandera as pa
import pydantic
import seaborn as sns
import xxhash
from napari.utils.notifications import show_info
from pandera.api.pandas.model import DataFrameModel
from pandera.typing import DataFrame, Index, Series


def join_index_columns(df: pd.DataFrame, index_columns: Sequence[str]) -> pd.Series:
    return (
        df.reset_index()
        .loc[:, list(index_columns)]
        .apply(lambda x: "_".join(map(str, x)), axis=1)
    ).rename("object_id")


def hash_single_object_id(object_id: str) -> float:
    max_value = 2**32
    return xxhash.xxh32(object_id).intdigest() / max_value


def get_normalized_hash_column(
    df: pd.DataFrame, index_columns: Sequence[str] = ("roi_id", "label")
) -> pd.Series:
    return join_index_columns(df, index_columns=index_columns).apply(
        hash_single_object_id
    )


def get_random_object_id(n_chars=10):
    return "".join(random.choice(string.ascii_lowercase) for i in range(n_chars))


def test_hash_function_leads_to_uniform_distribution():
    n = 80
    n_sites = 4
    n_per_site = n // n_sites
    site_ids = [
        e
        for e in chain.from_iterable(
            [[f"site{j}_{i}" for i in range(n_per_site)] for j in range(n_sites)]
        )
    ]
    random_ids = [get_random_object_id() for _ in range(n)]
    df1 = pd.Series(data=site_ids)
    df2 = pd.Series(data=random_ids)
    hash1 = df1.apply(hash_single_object_id)
    hash2 = df2.apply(hash_single_object_id)

    fig, ax = plt.subplots()
    sns.histplot(hash1, ax=ax)
    sns.histplot(hash2, ax=ax)


# %%
# TODO: This is cursed, annotations is supposed to be a nullable pandas int type but
# those don't seem to be recognized by pa.Check(ignore_na=True) when doing the checks :(
def get_input_and_internal_schemas(
    feature_names: Sequence[str],
    class_names: Sequence[str],
    index_columns: Sequence[str] = ("roi_id", "label"),
) -> tuple[pa.DataFrameSchema, pa.DataFrameSchema]:
    assert len(index_columns) == 2
    input_schema = pa.DataFrameSchema(
        columns={
            index_columns[0]: pa.Column(pa.String, coerce=True),
            index_columns[1]: pa.Column(pa.UInt64, coerce=True),
            "annotations": pa.Column(
                pa.Int64,
                coerce=True,
                # checks=pa.Check.between(1, len(class_names) + 1),
                checks=[
                    pa.Check(
                        lambda x: 1 <= x <= len(class_names),
                        element_wise=True,
                        ignore_na=True,
                    ),
                ],
                nullable=True,
            ),
            **{
                feature_name: pa.Column(pa.Float32, coerce=True)
                for feature_name in feature_names
            },
        },
        strict="filter",
        unique=list(index_columns),
    )

    internal_schema = input_schema.set_index(list(index_columns)).add_columns(
        {"hash": pa.Column(pa.Float32, coerce=True, checks=pa.Check.between(0.0, 1.0))}
    )
    return input_schema, internal_schema


# %%
index_columns = ["roi_id", "label"]
feature_names = ["f1", "f2"]
class_names = ["s", "m"]

input_schema, internal_schema = get_input_and_internal_schemas(
    feature_names,
    index_columns=index_columns,
    class_names=class_names,
)


# %%
n = 5
input_schema.validate(
    pd.DataFrame(
        {
            "roi_id": ["site1"] * n,
            "label": range(1, n + 1),
            "annotations": np.random.randint(1, len(class_names) + 1, size=n),
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
        }
    )
)
# %%


class Classifier:
    def __init__(self, feature_names, class_names):
        self._feature_names: list[str] = list(feature_names)
        self._class_names: list[str] = list(class_names)
        self._index_columns: list[str] = ["roi_id", "label"]
        self._input_schema, self._schema = get_input_and_internal_schemas(
            feature_names=feature_names,
            index_columns=self._index_columns,
            class_names=self._class_names,
        )
        # TODO: `self._schema.example(0)` does not return correct datatypes for
        # MultiIndex (issue: https://github.com/unionai-oss/pandera/issues/1049).
        # Can remove `self._schema.validate` call once fixed.
        self._data: pd.DataFrame = self._schema.validate(self._schema.example(0))

    def train(self):
        # TODO: Train the classifier
        show_info("Training classifier...")
        # TODO: Share training score

    def predict(self, df):
        # FIXME: Generate actual predictions for the df
        # FIXME: SettingWithCopyWarning => check if the actual run still
        # generates one, when we actually predict something
        with pd.option_context("mode.chained_assignment", None):
            df["predict"] = np.random.randint(1, 4, size=len(df))
        return df[["predict"]]

    def predict_on_dict(self, dict_of_dfs):
        # Make a prediction on each of the dataframes provided
        predicted_dicts = {}
        for roi in dict_of_dfs:
            show_info(f"Making a prediction for {roi=}...")
            predicted_dicts[roi] = self.predict(dict_of_dfs[roi])
        return predicted_dicts

    def add_features(self, df_raw: pd.DataFrame):
        # TODO: make sure objects with `annotation` == np.na get removed.
        show_info("Adding features...")
        #
        # Validate input
        df_valid_input = self._input_schema.validate(df_raw.reset_index())
        # Add hash column & set index
        df_valid_internal = df_valid_input.assign(
            hash=get_normalized_hash_column(df_valid_input, self._index_columns)
        ).set_index(self._index_columns)
        # Select index of rows not to be overwritten & update
        index = self._data.index.difference(df_valid_internal.index)
        self._data = pd.concat([self._data.loc[index], df_valid_internal]).sort_index()

    def add_dict_of_features(self, dict_of_features):
        # Add features for each roi
        # dict_of_features is a dict with roi as key & df as value
        for roi in dict_of_features:
            if 'roi_id' not in dict_of_features[roi]:
                df = dict_of_features[roi]['roid_id'] = roi
            else:
                df = df = dict_of_features[roi]
            show_info(f"Adding features for {roi=}...")
            self.add_features(df)

    def get_class_names(self):
        return self._class_names

    def get_feature_names(self):
        return self._feature_names

    def save(self, output_path):
        # TODO: Implement saving
        show_info(f"Saving classifier at {output_path}...")


# %%
feature_names = ["feature1", "feature2", "feature3"]
class_names = ["s", "m"]
df_raw = pd.read_csv(
    r"C:\Users\hessm\Documents\Programming\Python\fractal-napari-plugins\napari-feature-classifier\src\napari_feature_classifier\sample_data\test_df_with_roi.csv"
).assign(feature3=1)
df_raw = (
    df_raw.assign(
        annotations=np.random.randint(1, len(class_names) + 1, size=len(df_raw))
    )
    .assign(is_train=np.random.randint(2, size=len(df_raw)))
    .astype({"annotations": "Int64"})
)
df_raw.annotations.loc[:3] = None

df = df_raw.set_index(["roi_id", "label"])
df2 = (
    (df_raw.set_index("roi_id") + 17)
    .reset_index()
    .set_index(["roi_id", "label"])
    .subtract(17)
)
df3 = df_raw.drop("roi_id", axis=1).assign(roi_id="site2")
# %%
c = Classifier(["feature1", "feature2", "feature3"], ["s", "m"])
print(c._data)
c.add_features(df)
print(c._data)
c.add_features(df2)
print(c._data)
# c.add_features(df * 2)
# print(c._data)
c.add_features(df3)
print(c._data)
# print((c._data.hash < 0.2).sum() / len(c._data))

# %%
feature_names = ["feature1", "feature2", "feature3"]
input_schema = get_input_schema(feature_names)
schema = input_schema.set_index(["roi_id", "label"])
# %%
input_schema.validate(df.reset_index().reset_index())

# %%
schema = pa.DataFrameSchema(
    columns={
        "feature1": pa.Column(pa.Float32, coerce=True),
        "feature2": pa.Column(pa.Float32, coerce=True),
    },
    index=pa.MultiIndex(
        [
            pa.Index(pa.String, name="idx1", coerce=True),
            pa.Index(pa.Int64, name="idx2", coerce=True),
        ]
    ),
)

df = pd.DataFrame(
    data={
        "idx1": ["a", "b", "c"],
        "idx2": [0, 1, 2],
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.0, 6.0],
    }
).set_index(["idx1", "idx2"])

df_valid = schema.validate(df)

df_example = schema.example(3)
