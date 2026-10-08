from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_extraction import FEATURE_NAMES, extract_signal_features
from src.signal_generator import (
    MODULATIONS,
    SNR_LEVELS,
    generate_modulated_signal,
    generate_random_bits,
)
from src.utils import PROCESSED_DIR, ensure_directories


def generate_modulation_dataset(
    num_samples_per_modulation: int = 1000,
    snr_levels: tuple[int, ...] = SNR_LEVELS,
    output_path: str | None = None,
    samples_per_symbol_options: tuple[int, ...] = (4, 8, 16, 32),
) -> str:
    ensure_directories()
    if output_path is None:
        output_path = str(PROCESSED_DIR / 'modulation_dataset.csv')

    rng = np.random.default_rng(42)
    rows = []
    for modulation in MODULATIONS:
        for _ in range(num_samples_per_modulation):
            bits = generate_random_bits(256, rng=rng)
            snr_db = float(rng.choice(snr_levels))
            samples_per_symbol = int(rng.choice(samples_per_symbol_options))
            signal = generate_modulated_signal(
                modulation,
                bits=bits,
                snr_db=snr_db,
                rng=rng,
                samples_per_symbol=samples_per_symbol,
            )
            features = extract_signal_features(signal, samples_per_symbol=samples_per_symbol)
            row = dict(zip(FEATURE_NAMES, features))
            row['samples_per_symbol'] = samples_per_symbol
            row['label'] = modulation
            rows.append(row)

    dataframe = pd.DataFrame(rows, columns=FEATURE_NAMES + ['samples_per_symbol', 'label'])
    dataframe.to_csv(output_path, index=False)

    labels = dataframe['label'].value_counts()
    print('unique labels')
    print(sorted(dataframe['label'].unique()))
    print('samples per label')
    print(labels.to_string())
    print('\nFeature statistics by modulation:')
    for modulation in MODULATIONS:
        subset = dataframe[dataframe['label'] == modulation]
        print(f'\n{modulation}:')
        print(subset.describe().loc[['mean', 'std']].to_string())

    return output_path
