# FIXME: Get rid of show_info in classifier class
from napari.utils.notifications import show_info
import numpy as np

class Classifier(object):
    def __init__(self, feature_names, class_names):
        self._feature_names = feature_names
        self._class_names = class_names
        # TODO: Initialize the classifier

    def train(self):
        # TODO: Train the classifier
        show_info("Training classifier...")
        # TODO: Share training score

    def predict(self, df):
        # FIXME: Generate actual predictions for the df
        # FIXME: SettingWithCopyWarning
        df['predict'] = np.random.randint(1,4,size=len(df))
        return df[['predict']]
    
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
    
