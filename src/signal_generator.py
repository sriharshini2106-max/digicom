from __future__ import annotations

import numpy as np

MODULATIONS = ('BPSK', 'BFSK', 'QPSK', 'DPSK', '16-QAM')
SNR_LEVELS = (-10, -5, 0, 5, 10, 15, 20, 25, 30)
SAMPLES_PER_SYMBOL = 20
BFSK_F0 = 1.0 / (2.0 * SAMPLES_PER_SYMBOL)
BFSK_F1 = 1.0 / SAMPLES_PER_SYMBOL


def bfsk_frequencies(samples_per_symbol: int) -> tuple[float, float]:
    if samples_per_symbol < 1:
        raise ValueError('samples_per_symbol must be a positive integer.')
    return 1.0 / (2.0 * samples_per_symbol), 1.0 / samples_per_symbol


def generate_random_bits(length: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return rng.integers(0, 2, size=max(1, int(length)), dtype=np.int8)


def bpsk_modulate(bits: np.ndarray, samples_per_symbol: int = SAMPLES_PER_SYMBOL) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int8).ravel()
    symbols = np.where(bits == 1, 1.0 + 0.0j, -1.0 + 0.0j).astype(np.complex128)
    return np.repeat(symbols, samples_per_symbol)


def bfsk_modulate(
    bits: np.ndarray,
    f0: float | None = None,
    f1: float | None = None,
    samples_per_symbol: int = SAMPLES_PER_SYMBOL,
) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if bits.size == 0:
        return np.array([], dtype=np.complex128)

    if f0 is None or f1 is None:
        f0, f1 = bfsk_frequencies(samples_per_symbol)
    Fs = 1.0
    symbols = []
    for bit in bits:
        frequency = f1 if bit == 1 else f0
        n = np.arange(samples_per_symbol, dtype=np.float64)
        symbols.append(np.cos(2.0 * np.pi * frequency * n / Fs))
    return np.concatenate(symbols).astype(np.complex128)


def qpsk_modulate(bits: np.ndarray, samples_per_symbol: int = SAMPLES_PER_SYMBOL) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if bits.size % 2 != 0:
        bits = np.pad(bits, (0, 1), constant_values=0)

    phase_map = {
        0: np.pi / 4,
        1: 3 * np.pi / 4,
        3: 5 * np.pi / 4,
        2: 7 * np.pi / 4,
    }
    symbols = []
    for pair in bits.reshape(-1, 2):
        index = int(pair[0]) * 2 + int(pair[1])
        symbols.append(np.exp(1j * phase_map[index]))
    symbols = np.asarray(symbols, dtype=np.complex128)
    return np.repeat(symbols, samples_per_symbol)


def qam_modulate(bits: np.ndarray, samples_per_symbol: int = SAMPLES_PER_SYMBOL) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if bits.size % 4 != 0:
        pad = 4 - (bits.size % 4)
        bits = np.pad(bits, (0, pad), constant_values=0)

    gray_levels = np.array([-3.0, -1.0, 3.0, 1.0], dtype=np.float64) / np.sqrt(10.0)
    lookup = np.empty(16, dtype=np.complex128)
    for code in range(16):
        i_bits = (code >> 2) & 0b11
        q_bits = code & 0b11
        lookup[code] = gray_levels[i_bits] + 1j * gray_levels[q_bits]

    symbols = []
    for chunk in bits.reshape(-1, 4):
        code = int(''.join(chunk.astype(str)), 2)
        symbols.append(lookup[code])
    symbols = np.asarray(symbols, dtype=np.complex128)
    return np.repeat(symbols, samples_per_symbol)


def dpsk_modulate(bits: np.ndarray, samples_per_symbol: int = SAMPLES_PER_SYMBOL) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if bits.size == 0:
        return np.array([], dtype=np.complex128)

    phase = 0.0
    symbols = []
    for bit in bits:
        delta_phase = 0.0 if bit == 0 else np.pi
        phase = phase + delta_phase
        symbols.append(np.exp(1j * phase))
    symbols = np.asarray(symbols, dtype=np.complex128)
    return np.repeat(symbols, samples_per_symbol)


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    signal = np.asarray(signal, dtype=np.complex128)
    signal_power = float(np.mean(np.abs(signal) ** 2))
    if signal_power <= 1e-12:
        signal_power = 1.0
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape))
    return signal + noise


def generate_modulated_signal(
    modulation: str,
    bits: np.ndarray | None = None,
    snr_db: float | None = None,
    rng: np.random.Generator | None = None,
    samples_per_symbol: int = SAMPLES_PER_SYMBOL,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    modulation = '16-QAM' if modulation in {'QAM', '16QAM'} else modulation
    if modulation not in MODULATIONS:
        raise ValueError(f'Unsupported modulation: {modulation}')

    if bits is None:
        bits = generate_random_bits(256, rng=rng)
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if samples_per_symbol < 1:
        raise ValueError('samples_per_symbol must be a positive integer.')

    if modulation == 'BPSK':
        signal = bpsk_modulate(bits, samples_per_symbol)
    elif modulation == 'BFSK':
        signal = bfsk_modulate(bits=bits, samples_per_symbol=samples_per_symbol)
    elif modulation == 'QPSK':
        signal = qpsk_modulate(bits, samples_per_symbol)
    elif modulation == 'DPSK':
        signal = dpsk_modulate(bits, samples_per_symbol)
    elif modulation == '16-QAM':
        signal = qam_modulate(bits, samples_per_symbol)
    else:
        raise ValueError(f'Unsupported modulation: {modulation}')

    if snr_db is not None:
        signal = add_awgn(signal, snr_db, rng=rng)
    return signal
