from collections import OrderedDict
from zlib import crc32
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from pathlib import Path
import pickle


def main():
    pass


def load_features(fld, structure=None, index_col=("filename_prefix", "Label"), glob_str="*.csv"):
    fld = Path(fld)
    features = []
    for fn in fld.glob(glob_str):
        print(fn.name)
        features.append(pd.read_csv(fn))
    df = pd.concat(features, axis=0, ignore_index=True)
    if structure:
        df = df[df["structure"] == structure]
    df = df.set_index(list(index_col))
    return df


def read_feather(fn, index_col=("filename_prefix", "Label")):
    df = pd.read_feather(fn)
    if index_col:
        df = df.set_index(list(index_col))
    return df


def make_identifier(df):
    str_id = df.apply(lambda x: "_".join(map(str, x)), axis=1)
    return str_id


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(hash(identifier))) & 0xFFFFFFFF < test_ratio * 2 ** 32


def load_classifier(path):
    with open(path, 'rb') as f:
        return pickle.loads(f.read())


class Classifier:
    def __init__(self, name, features, training_features=None, index_columns=None):
        self.name = name
        self.clf = RandomForestClassifier()
        full_data = features
        full_data.loc[:, "train"] = 0
        full_data.loc[:, "predict"] = 0
        self.index_columns = index_columns
        self.train_data = full_data[["train"]]
        self.predict_data = full_data[["predict"]]
        self.bbx = full_data[
            [k for k in full_data.keys() if k.startswith("BoundingBox")]
        ]
        # self.positions = full_data[['midpoints_z', 'midpoints_y', 'midpoints_x']]
        if training_features is None:
            self.training_features = {
                "Centroid_z",
                "Elongation",
                "FeretDiameter",
                "Flatness",
                "NumberOfPixels",
                "NumberOfPixelsOnBorder",
                "Perimeter",
                "PerimeterOnBorder",
                "Roundness",
                "PrincipalMoments_x",
                "PrincipalMoments_y",
                "PrincipalMoments_z",
                "DAPI0_Mean",
                "DAPI0_Median",
                "DAPI0_Skewness",
                "DAPI0_Variance",
                "DAPI0_Kurtosis",
                "bCatenin0_Mean",
                "bCatenin0_Median",
                "bCatenin0_Skewness",
                "bCatenin0_Variance",
                "bCatenin0_Kurtosis",
                "PCNA0_Mean",
                "PCNA0_Median",
                "PCNA0_Skewness",
                "PCNA0_Variance",
                "PCNA0_Kurtosis",
                "NTouchingNeighbors",
                "Density_0.05",
                "Density_0.1",
                "Density_0.2",
                "Density_0.3",
                "Density_0.5",
            }
        elif training_features == "all":
            self.training_features = set(
                full_data.select_dtypes([np.number])
                    .drop(["train", "predict"], axis=1, errors="ignore")
                    .keys()
            )
        else:
            self.training_features = training_features
        self.data = full_data[self.training_features]

    @staticmethod
    def train_test_split(df, test_perc=0.2, index_columns=None):
        in_test_set = make_identifier(df.reset_index()[list(index_columns)]).apply(
            test_set_check, args=(test_perc,)
        )
        return df.iloc[~in_test_set.values, :], df.iloc[in_test_set.values, :]

    def train(self):
        X_train, X_test = self.train_test_split(
            self.data[self.train_data["train"] > 0], index_columns=self.index_columns
        )
        y_train, y_test = self.train_test_split(
            self.train_data[self.train_data["train"] > 0], index_columns=self.index_columns
        )
        assert np.all(X_train.index == y_train.index)
        assert np.all(X_test.index == y_test.index)
        print(
            "Annotations split into {} training and {} test samples...".format(
                len(X_train), len(X_test)
            )
        )
        self.clf.fit(X_train, y_train)

        print(
            "F1 score on test set: {}".format(
                f1_score(y_test, self.clf.predict(X_test), average="macro")
            )
        )
        self.predict_data.loc[:] = self.clf.predict(self.data).reshape(-1, 1)
        print("done")

    def predict(self, data):
        data = data[self.training_features]
        return self.clf.predict(data)

    def feature_importance(self):
        return OrderedDict(
            sorted(
                {
                    f: i
                    for f, i in zip(
                    self.training_features, self.clf.feature_importances_
                )
                }.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    def most_important(self, n=5):
        return list(self.feature_importance().keys())[:n]

    def save(self):
        s = pickle.dumps(self)
        with open(self.name + ".clf", "wb") as f:
            f.write(s)


if __name__ == "__main__":
    main()
