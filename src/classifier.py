from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.feature_extraction import FEATURE_NAMES, extract_signal_features


class ModulationClassifier:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parents[1] / 'models' / 'modulation_classifier.pkl'
        self.bundle = joblib.load(self.model_path)
        self.model = self.bundle['model']
        self.scaler = self.bundle['scaler']
        self.feature_names = self.bundle.get('feature_names', FEATURE_NAMES)
        self.labels = self.bundle.get('labels', ['BPSK', 'BFSK', 'QPSK', 'DPSK', '16-QAM'])

    def predict_with_confidence(self, signal, samples_per_symbol: int = 20):
        features = extract_signal_features(signal, samples_per_symbol=samples_per_symbol)
        if len(features) != len(self.feature_names):
            raise ValueError(f'Feature length mismatch. Expected {len(self.feature_names)} features but got {len(features)}.')
        sample = np.asarray([features], dtype=np.float64)
        sample = self.scaler.transform(sample)
        prediction = self.model.predict(sample)[0]
        probabilities = self.model.predict_proba(sample)[0]
        return prediction, dict(zip(self.model.classes_, probabilities))

    def predict(self, signal, samples_per_symbol: int = 20):
        prediction, _ = self.predict_with_confidence(signal, samples_per_symbol)
        return prediction
