from __future__ import annotations

import numpy as np


def bit_accuracy(original_bits: np.ndarray, recovered_bits: np.ndarray) -> float:
    original_bits = np.asarray(original_bits, dtype=np.int8).ravel()
    recovered_bits = np.asarray(recovered_bits, dtype=np.int8).ravel()
    min_length = min(original_bits.size, recovered_bits.size)
    if min_length == 0:
        return 0.0
    return float(np.mean(original_bits[:min_length] == recovered_bits[:min_length]))


def bit_error_rate(original_bits: np.ndarray, recovered_bits: np.ndarray) -> float:
    original_bits = np.asarray(original_bits, dtype=np.int8).ravel()
    recovered_bits = np.asarray(recovered_bits, dtype=np.int8).ravel()
    min_length = min(original_bits.size, recovered_bits.size)
    if min_length == 0:
        return 0.0
    diff = np.sum(original_bits[:min_length] != recovered_bits[:min_length])
    return float(diff / min_length)


def compare_known_bits(
    original_bits: np.ndarray,
    recovered_bits: np.ndarray,
    modulation: str,
) -> tuple[int, int, float, float]:
    original_bits = np.asarray(original_bits, dtype=np.int8).ravel()
    recovered_bits = np.asarray(recovered_bits, dtype=np.int8).ravel()
    if modulation == 'DPSK':
        original_bits = original_bits[1:]
    compared = min(original_bits.size, recovered_bits.size)
    if compared == 0:
        return 0, 0, float('nan'), float('nan')
    errors = int(np.count_nonzero(original_bits[:compared] != recovered_bits[:compared]))
    ber = float(errors / compared)
    return errors, compared, ber, 1.0 - ber


def evaluate_signal(original_bits: np.ndarray, signal: np.ndarray, modulation: str, demodulator) -> dict:
    recovered_bits = demodulator(signal, modulation)
    return {
        'modulation': modulation,
        'accuracy': bit_accuracy(original_bits, recovered_bits),
        'bit_error_rate': bit_error_rate(original_bits, recovered_bits),
    }
