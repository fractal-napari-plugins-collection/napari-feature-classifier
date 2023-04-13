# FIXME: Get rid of show_info in classifier class
from napari.utils.notifications import show_info
import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

class ClassifierDataSchema(pa.SchemaModel):
    # TODO: Get
    # FIXME: Make those index columns
    roi_id: Series[str] = pa.Field(coerce=True, unique=False)
    label: Series[int] = pa.Field(coerce=True, unique=True)


class Classifier(object):
    def __init__(self, feature_names, class_names):
        self._feature_names = feature_names
        self._class_names = class_names
        self._label_column = "label"
        self._roi_id_colum = "roi_id"
        # TODO: Set up data storage schema
        self.classifier_data = DataFrame[ClassifierDataSchema]()

        # TODO: Initialize the classifier

    def train(self):
        # TODO: Train the classifier
        show_info("Training classifier...")
        # TODO: Share training score

    def predict(self, df):
        # FIXME: Generate actual predictions for the df
        # FIXME: SettingWithCopyWarning => check if the actual run still 
        # generates one, when we actually predict something
        with pd.option_context('mode.chained_assignment', None):
            df["predict"] = np.random.randint(1, 4, size=len(df))
        return df[["predict"]]

    def predict_on_dict(self, dict_of_dfs):
        # Make a prediction on each of the dataframes provided
        predicted_dicts = {}
        for roi in dict_of_dfs:
            show_info(f"Making a prediction for {roi=}...")
            predicted_dicts[roi] = self.predict(dict_of_dfs[roi])
        return predicted_dicts

    def add_features(self, new_feature_df):
        # TODO: Add features
        show_info("Adding features...")

    def get_class_names(self):
        return self._class_names

    def get_feature_names(self):
        return self._feature_names

    def save(self, output_path):
        # TODO: Implement saving
        show_info(f"Saving classifier at {output_path}...")
