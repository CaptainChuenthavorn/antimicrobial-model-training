import numpy as np

class SMOTERounding:
    def __init__(self, smote) -> None:
        if hasattr(smote, "fit_resample"):
            self.smote = smote
        else:
            raise Exception("Method fit_resample not found.")

    def fit_resample(self, X, y):
        X_resample, y_resample = self.smote.fit_resample(
            X.astype(np.float64), y)
        X_resample = np.round(X_resample).astype(np.int32)
        return X_resample, y_resample