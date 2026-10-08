from __future__ import annotations

import numpy as np
import pandas as pd

from src.classifier import ModulationClassifier
from src.dataset_generator import generate_modulation_dataset
from src.demodulator import demodulate_signal
from src.evaluation import compare_known_bits
from src.signal_generator import MODULATIONS, SNR_LEVELS, generate_modulated_signal, generate_random_bits
from src.train_model import DATASET_PATH, MODEL_PATH, train_and_save_model
from src.utils import METRICS_DIR

EVALUATION_REPETITIONS = 10
EVALUATION_SAMPLES_PER_SYMBOL = 16


def evaluate_system_for_snr(
    snr_db: float,
    modulation: str,
    classifier: ModulationClassifier,
    rng: np.random.Generator,
) -> dict:
    bits = generate_random_bits(256, rng=rng)
    signal = generate_modulated_signal(
        modulation,
        bits=bits,
        snr_db=float(snr_db),
        rng=rng,
        samples_per_symbol=EVALUATION_SAMPLES_PER_SYMBOL,
    )

    predicted, probabilities = classifier.predict_with_confidence(
        signal,
        samples_per_symbol=EVALUATION_SAMPLES_PER_SYMBOL,
    )
    automatic_bits = demodulate_signal(
        signal,
        predicted,
        samples_per_symbol=EVALUATION_SAMPLES_PER_SYMBOL,
    )
    oracle_bits = demodulate_signal(
        signal,
        modulation,
        samples_per_symbol=EVALUATION_SAMPLES_PER_SYMBOL,
    )
    automatic_errors, automatic_count, automatic_ber, automatic_accuracy = compare_known_bits(
        bits,
        automatic_bits,
        modulation,
    )
    oracle_errors, oracle_count, oracle_ber, oracle_accuracy = compare_known_bits(
        bits,
        oracle_bits,
        modulation,
    )
    return {
        'snr_db': snr_db,
        'actual_modulation': modulation,
        'predicted_modulation': predicted,
        'classification_correct': predicted == modulation,
        'predicted_probability': float(probabilities[predicted]),
        'samples_per_symbol': EVALUATION_SAMPLES_PER_SYMBOL,
        'automatic_bit_errors': automatic_errors,
        'automatic_compared_bits': automatic_count,
        'automatic_ber': automatic_ber,
        'automatic_demodulation_accuracy': automatic_accuracy,
        'oracle_bit_errors': oracle_errors,
        'oracle_compared_bits': oracle_count,
        'oracle_ber': oracle_ber,
        'oracle_demodulation_accuracy': oracle_accuracy,
    }


def main() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        generate_modulation_dataset(output_path=str(DATASET_PATH))
    train_and_save_model(dataset_path=str(DATASET_PATH), model_path=str(MODEL_PATH))
    classifier = ModulationClassifier(MODEL_PATH)

    rng = np.random.default_rng()
    rows = [
        evaluate_system_for_snr(snr_db, modulation, classifier, rng)
        for modulation in MODULATIONS
        for snr_db in SNR_LEVELS
        for _ in range(EVALUATION_REPETITIONS)
    ]
    results = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METRICS_DIR / 'test_summary.csv'
    results.to_csv(output_path, index=False)

    summary = (
        results.groupby(['snr_db', 'actual_modulation'], sort=True)
        .agg(
            classification_accuracy=('classification_correct', 'mean'),
            automatic_demodulation_accuracy=('automatic_demodulation_accuracy', 'mean'),
            automatic_ber=('automatic_ber', 'mean'),
            oracle_demodulation_accuracy=('oracle_demodulation_accuracy', 'mean'),
            oracle_ber=('oracle_ber', 'mean'),
        )
    )
    print(summary.to_string())
    print(f'\nSaved {len(results)} independently generated trials to {output_path}')
    return results


if __name__ == '__main__':
    main()
