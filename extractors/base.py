from PIL import Image


class FeatureExtractor:
    type = 'base'

    def __init__(self, input, **kwargs):
        self.input_directory = input

    def extract_features(self, image: Image):
        raise NotImplementedError('base classes should implement this function')
