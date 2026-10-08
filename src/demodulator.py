from __future__ import annotations

import numpy as np

from src.signal_generator import MODULATIONS, SAMPLES_PER_SYMBOL, bfsk_frequencies


def bpsk_demodulate(signal: np.ndarray, samples_per_symbol: int = SAMPLES_PER_SYMBOL) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128).ravel()
    bits = []
    for idx in range(0, signal.size, samples_per_symbol):
        segment = signal[idx:idx + samples_per_symbol]
        if segment.size == 0:
            continue
        if segment.size < samples_per_symbol:
            segment = np.pad(segment, (0, samples_per_symbol - segment.size), constant_values=0)
        y = np.real(segment[0])
        bits.append(1 if y >= 0 else 0)
    return np.asarray(bits, dtype=np.int8)


def bfsk_demodulate(
    signal: np.ndarray,
    f0: float | None = None,
    f1: float | None = None,
    samples_per_symbol: int = SAMPLES_PER_SYMBOL,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128).ravel()
    bits = []
    if f0 is None or f1 is None:
        f0, f1 = bfsk_frequencies(samples_per_symbol)
    Fs = 1.0
    for idx in range(0, signal.size, samples_per_symbol):
        segment = signal[idx:idx + samples_per_symbol]
        if segment.size < samples_per_symbol:
            segment = np.pad(segment, (0, samples_per_symbol - segment.size), constant_values=0)
        n = np.arange(segment.size, dtype=np.float64)
        c0 = np.cos(2.0 * np.pi * f0 * n / Fs).astype(np.float64)
        c1 = np.cos(2.0 * np.pi * f1 * n / Fs).astype(np.float64)
        E0 = np.abs(np.sum(np.real(segment) * c0)) ** 2
        E1 = np.abs(np.sum(np.real(segment) * c1)) ** 2
        bits.append(1 if E1 > E0 else 0)
    return np.asarray(bits, dtype=np.int8)


def qpsk_demodulate(signal: np.ndarray, samples_per_symbol: int = SAMPLES_PER_SYMBOL) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128).ravel()
    phase_reference = np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4], dtype=np.float64)
    bit_lookup = {
        0: np.array([0, 0], dtype=np.int8),
        1: np.array([0, 1], dtype=np.int8),
        2: np.array([1, 1], dtype=np.int8),
        3: np.array([1, 0], dtype=np.int8),
    }
    bits = []
    for idx in range(0, signal.size, samples_per_symbol):
        segment = signal[idx:idx + samples_per_symbol]
        if segment.size < samples_per_symbol:
            segment = np.pad(segment, (0, samples_per_symbol - segment.size), constant_values=0)
        symbol = np.mean(segment)
        theta = np.angle(symbol)
        index = int(np.argmin(np.abs(np.angle(np.exp(1j * (theta - phase_reference))))))
        bits.extend(bit_lookup[index])
    return np.asarray(bits, dtype=np.int8)


def qam_demodulate(signal: np.ndarray, samples_per_symbol: int = SAMPLES_PER_SYMBOL) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128).ravel()
    levels = np.array([-3.0, -1.0, 3.0, 1.0], dtype=np.float64)
    bit_map = {
        -3.0: np.array([0, 0], dtype=np.int8),
        -1.0: np.array([0, 1], dtype=np.int8),
        3.0: np.array([1, 0], dtype=np.int8),
        1.0: np.array([1, 1], dtype=np.int8),
    }
    recovered_bits = []
    for idx in range(0, signal.size, samples_per_symbol):
        segment = signal[idx:idx + samples_per_symbol]
        if segment.size < samples_per_symbol:
            segment = np.pad(segment, (0, samples_per_symbol - segment.size), constant_values=0)
        symbol = np.mean(segment) * np.sqrt(10.0)
        i_hat = levels[np.argmin(np.abs(levels - np.real(symbol)))]
        q_hat = levels[np.argmin(np.abs(levels - np.imag(symbol)))]
        recovered_bits.extend(bit_map[float(i_hat)].tolist())
        recovered_bits.extend(bit_map[float(q_hat)].tolist())
    return np.asarray(recovered_bits, dtype=np.int8)


def dpsk_demodulate(signal: np.ndarray, samples_per_symbol: int = SAMPLES_PER_SYMBOL) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.complex128).ravel()
    if signal.size < samples_per_symbol:
        return np.array([], dtype=np.int8)

    symbols = []
    for idx in range(0, signal.size, samples_per_symbol):
        segment = signal[idx:idx + samples_per_symbol]
        if segment.size < samples_per_symbol:
            segment = np.pad(segment, (0, samples_per_symbol - segment.size), constant_values=0)
        symbols.append(np.mean(segment))
    symbols = np.asarray(symbols, dtype=np.complex128)
    if symbols.size < 2:
        return np.array([], dtype=np.int8)

    z = symbols[1:] * np.conj(symbols[:-1])
    delta_theta = np.angle(z)
    recovered = (np.cos(delta_theta) < 0.0).astype(np.int8)
    return recovered


def demodulate_signal(
    signal: np.ndarray,
    modulation: str,
    samples_per_symbol: int = SAMPLES_PER_SYMBOL,
) -> np.ndarray:
    modulation = '16-QAM' if modulation in {'QAM', '16QAM'} else modulation
    if modulation not in MODULATIONS:
        raise ValueError(f'Unsupported modulation: {modulation}')
    if modulation == 'BPSK':
        return bpsk_demodulate(signal, samples_per_symbol)
    if modulation == 'BFSK':
        return bfsk_demodulate(samples_per_symbol=samples_per_symbol, signal=signal)
    if modulation == 'QPSK':
        return qpsk_demodulate(signal, samples_per_symbol)
    if modulation == 'DPSK':
        return dpsk_demodulate(signal, samples_per_symbol)
    if modulation == '16-QAM':
        return qam_demodulate(signal, samples_per_symbol)
    raise ValueError(f'Unsupported modulation: {modulation}')
