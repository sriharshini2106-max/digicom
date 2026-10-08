from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.dataset_generator import generate_modulation_dataset
from src.feature_extraction import FEATURE_NAMES
from src.utils import MODELS_DIR, PROCESSED_DIR

DATASET_PATH = PROCESSED_DIR / 'modulation_dataset.csv'
MODEL_PATH = MODELS_DIR / 'modulation_classifier.pkl'


def train_and_save_model(dataset_path: str | None = None, model_path: str | None = None) -> dict:
    dataset_path = str(DATASET_PATH if dataset_path is None else dataset_path)
    model_path = str(MODEL_PATH if model_path is None else model_path)

    dataframe = pd.read_csv(dataset_path)
    if dataframe.empty:
        raise ValueError('The modulation dataset is empty.')

    labels = sorted(dataframe['label'].unique().tolist())
    if set(labels) != set(['BPSK', 'BFSK', 'QPSK', 'DPSK', '16-QAM']):
        raise ValueError(f'Unexpected labels in dataset: {labels}')

    features = [column for column in FEATURE_NAMES if column in dataframe.columns]
    X = dataframe[features].to_numpy(dtype=np.float64)
    y = dataframe['label'].to_numpy()

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(splitter.split(X, y))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight='balanced_subsample',
        min_samples_leaf=1,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    cm = confusion_matrix(y_test, predictions, labels=labels)

    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_names': features,
        'labels': labels,
        'samples_per_symbol_options': (4, 8, 16, 32),
        'default_samples_per_symbol': 16,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
    }
    joblib.dump(bundle, model_path)

    print('Training labels:', labels)
    print('Train/test split: 80/20 with stratification')
    print('Hold-out accuracy:', accuracy)
    print(classification_report(y_test, predictions, target_names=labels))
    return bundle


def main() -> dict:
    dataset_path = str(DATASET_PATH)
    if not DATASET_PATH.exists():
        dataset_path = generate_modulation_dataset(output_path=str(DATASET_PATH))
    return train_and_save_model(dataset_path=dataset_path)


if __name__ == '__main__':
    main()
