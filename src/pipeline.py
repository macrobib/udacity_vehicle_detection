

class pipeline:
    """Implementing the overall detection and training pipelines."""
    def __init__(self, classifier="svm"):
        self.classifier = classifier

    def train(self):
        """Training pipeline"""
        pass

    def validate(self):
        """Validation pipeline"""
        pass

    def predict(self):
        """Prediction pipeline."""
        pass
