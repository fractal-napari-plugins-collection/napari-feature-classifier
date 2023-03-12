# FIXME: Get rid of show_info in classifier class
from napari.utils.notifications import show_info

class Classifier(object):
    def __init__(self, feature_names, class_names):
        self._feature_names = feature_names
        self._class_names = class_names
        # TODO: Initialize the classifier

    def train(self):
        # TODO: Train the classifier
        show_info("Training classifier...")

    def predict(self):
        # TODO: Make a prediction
        show_info("Making a prediction...")

    def add_features(self, new_feature_df):
        # TODO: Add features
        show_info("Adding features...")
    
    def get_class_names(self):
        return self._class_names
    
    def save(self, output_path):
        # TODO: Implement saving
        show_info(f"Saving classifier at {output_path}...")
    
