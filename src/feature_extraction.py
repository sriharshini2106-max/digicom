from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    'mean_amplitude', 'rms_amplitude', 'amplitude_variance', 'amplitude_std',
    'peak_amplitude', 'crest_factor', 'zero_crossing_rate',
    'mean_I', 'mean_Q', 'variance_I', 'variance_Q', 'covariance_IQ', 'correlation_IQ',
    'mean_envelope', 'envelope_variance', 'envelope_std', 'envelope_kurtosis',
    'mean_phase', 'phase_variance', 'phase_std', 'mean_phase_diff', 'phase_diff_variance',
    'phase_diff_std', 'mean_inst_frequency', 'std_inst_frequency', 'dominant_inst_frequency',
    'dominant_frequency', 'spectral_centroid', 'spectral_bandwidth', 'spectral_entropy',
    'spectral_variance', 'peak_spectral_magnitude', 'significant_spectral_peaks',
    'skewness', 'kurtosis', 'cumulant_4th_order',
    'bpsk_symbol_confidence', 'dpsk_phase_confidence', 'direct_vs_differential_gap'
]


def safe_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def zero_crossing_rate(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size < 2:
        return 0.0
    signs = np.sign(signal)
    signs[signs == 0] = 1
    return float(np.mean(np.abs(np.diff(signs)) > 0))


def skewness(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size < 2:
        return 0.0
    mean = np.mean(values)
    std = safe_std(values)
    if std < 1e-12:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 3))


def kurtosis(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size < 2:
        return 0.0
    mean = np.mean(values)
    std = safe_std(values)
    if std < 1e-12:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 4))


def instantaneous_frequency(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.complex128).ravel()
    if signal.size < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    phase = np.unwrap(np.angle(signal))
    diffs = np.diff(phase)
    inst_freq = (1.0 / (2.0 * np.pi)) * diffs
    return phase, inst_freq


def extract_signal_features(signal: np.ndarray, samples_per_symbol: int = 20) -> list[float]:
    signal = np.asarray(signal, dtype=np.complex128).ravel()
    if signal.size == 0:
        return [0.0] * len(FEATURE_NAMES)

    real = np.real(signal)
    imag = np.imag(signal)
    magnitude = np.abs(signal)
    phase = np.angle(signal)
    phase_unwrapped = np.unwrap(phase)
    phase_diff = np.diff(phase_unwrapped)
    inst_freq = (1.0 / (2.0 * np.pi)) * phase_diff
    real_signal = np.real(signal)
    fft_mag = np.abs(np.fft.rfft(real_signal))
    freqs = np.fft.rfftfreq(real_signal.size, d=1.0)
    if fft_mag.size == 0:
        fft_mag = np.array([0.0])
        freqs = np.array([0.0])
    if np.sum(fft_mag) < 1e-12:
        dominant_freq = 0.0
        centroid = 0.0
        bandwidth = 0.0
        entropy = 0.0
        spectral_var = 0.0
        peak_mag = 0.0
        significant_peaks = 0
    else:
        normalized = fft_mag / (np.sum(fft_mag) + 1e-12)
        dominant_freq = float(freqs[np.argmax(fft_mag)])
        centroid = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-12))
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_mag) / (np.sum(fft_mag) + 1e-12)))
        entropy = float(-np.sum(normalized * np.log2(normalized + 1e-12)))
        spectral_var = float(np.var(fft_mag))
        peak_mag = float(np.max(fft_mag))
        significant = fft_mag > 0.1 * np.max(fft_mag)
        significant_peaks = int(np.sum(significant))

    mean_abs = float(np.mean(magnitude))
    rms = float(np.sqrt(np.mean(magnitude ** 2)))
    amp_var = float(np.var(magnitude))
    amp_std = float(np.std(magnitude))
    peak_amp = float(np.max(magnitude))
    crest = float(peak_amp / (rms + 1e-12))

    mean_i = float(np.mean(real))
    mean_q = float(np.mean(imag))
    var_i = float(np.var(real))
    var_q = float(np.var(imag))
    covariance = float(np.cov(real, imag)[0, 1]) if real.size > 1 else 0.0
    corr = 0.0
    if var_i > 1e-12 and var_q > 1e-12:
        corr = float(covariance / (np.sqrt(var_i * var_q) + 1e-12))

    env_mean = float(np.mean(magnitude))
    env_var = float(np.var(magnitude))
    env_std = float(np.std(magnitude))
    env_kurt = float(kurtosis(magnitude))

    phase_mean = float(np.mean(phase))
    phase_var = float(np.var(phase))
    phase_std = float(np.std(phase))
    if phase_diff.size > 0:
        mean_phase_diff = float(np.mean(phase_diff))
        phase_diff_variance = float(np.var(phase_diff))
        phase_diff_std = float(np.std(phase_diff))
    else:
        mean_phase_diff = 0.0
        phase_diff_variance = 0.0
        phase_diff_std = 0.0

    if inst_freq.size > 0:
        mean_inst_freq = float(np.mean(inst_freq))
        std_inst_freq = float(np.std(inst_freq))
        dominant_inst = float(np.max(np.abs(inst_freq)))
    else:
        mean_inst_freq = 0.0
        std_inst_freq = 0.0
        dominant_inst = 0.0

    if samples_per_symbol < 1:
        raise ValueError('samples_per_symbol must be a positive integer.')
    segments = []
    for idx in range(0, signal.size, samples_per_symbol):
        segment = signal[idx:idx + samples_per_symbol]
        if segment.size == 0:
            continue
        if segment.size < samples_per_symbol:
            segment = np.pad(segment, (0, samples_per_symbol - segment.size), constant_values=0)
        segments.append(segment)
    if segments:
        real_means = np.asarray([np.mean(np.real(segment)) for segment in segments], dtype=np.float64)
        phase_groups = np.asarray([np.angle(np.mean(segment)) for segment in segments], dtype=np.float64)
        bpsk_decisions = (real_means >= 0.0).astype(np.int8)
        dpsk_decisions = (np.abs(np.diff(phase_groups)) > np.pi / 2).astype(np.int8)
        bpsk_symbol_confidence = float(np.mean(np.abs(real_means)))
        dpsk_phase_confidence = float(np.mean(np.abs(np.abs(np.diff(phase_groups)) - np.pi))) if phase_groups.size > 1 else 0.0
        direct_vs_differential_gap = float(np.mean(bpsk_decisions[:len(dpsk_decisions)] != dpsk_decisions)) if len(dpsk_decisions) > 0 else 0.0
    else:
        bpsk_symbol_confidence = 0.0
        dpsk_phase_confidence = 0.0
        direct_vs_differential_gap = 0.0

    feature_vector = [
        mean_abs,
        rms,
        amp_var,
        amp_std,
        peak_amp,
        crest,
        zero_crossing_rate(real),
        mean_i,
        mean_q,
        var_i,
        var_q,
        covariance,
        corr,
        env_mean,
        env_var,
        env_std,
        env_kurt,
        phase_mean,
        phase_var,
        phase_std,
        mean_phase_diff,
        phase_diff_variance,
        phase_diff_std,
        mean_inst_freq,
        std_inst_freq,
        dominant_inst,
        dominant_freq,
        centroid,
        bandwidth,
        entropy,
        spectral_var,
        peak_mag,
        significant_peaks,
        float(skewness(real)),
        float(kurtosis(real)),
        float(kurtosis(magnitude)),
        bpsk_symbol_confidence,
        dpsk_phase_confidence,
        direct_vs_differential_gap,
    ]
    return feature_vector
