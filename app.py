from __future__ import annotations

import hashlib
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.classifier import ModulationClassifier
from src.demodulator import demodulate_signal
from src.evaluation import compare_known_bits
from src.signal_csv import load_signal_csv, signal_to_csv_bytes
from src.signal_generator import (
    MODULATIONS,
    SNR_LEVELS,
    bfsk_frequencies,
    generate_modulated_signal,
    generate_random_bits,
)
from src.train_model import MODEL_PATH, train_and_save_model
from src.utils import METRICS_DIR

BITS_PER_SYMBOL = {'BPSK': 1, 'BFSK': 1, 'QPSK': 2, 'DPSK': 1, '16-QAM': 4}
SAMPLES_PER_SYMBOL_OPTIONS = (4, 8, 16, 32)
BIT_LENGTH_OPTIONS = (1000, 2000, 5000, 10000)
EVALUATION_SAMPLES_PER_SYMBOL = 16
EVALUATION_PATH = METRICS_DIR / 'dashboard_evaluation.csv'
CLASS_ORDER = ('BPSK', 'BFSK', 'QPSK', 'DPSK', '16-QAM')
DEMODULATION_METHODS = {
    'BPSK': 'BPSK detection',
    'BFSK': 'Frequency correlation',
    'QPSK': 'Coherent QPSK detection',
    'DPSK': 'Differential detection',
    '16-QAM': 'I/Q symbol detection',
}

st.set_page_config(
    page_title='Automatic Modulation Recognition',
    layout='wide',
)

st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp { background: #f5f7fa; color: #172b45; }
    [data-testid="stHeader"] { background: #f5f7fa; }
    [data-testid="stSidebar"] {
        background: #eef2f6;
        border-right: 1px solid #d5dde7;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #203b5c !important;
    }
    h1, h2, h3, h4, p, label { color: #172b45; }
    h1 { font-size: 1.8rem !important; margin-bottom: 0 !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.05rem !important; }
    .subtitle { color: #52677e; font-size: .96rem; margin: .15rem 0 .85rem; }
    .section-title {
        color: #203b5c; font-size: .88rem; font-weight: 700;
        letter-spacing: .045em; text-transform: uppercase;
        border-bottom: 1px solid #d2dce7; padding-bottom: .4rem;
        margin: 1.2rem 0 .7rem;
    }
    div[data-testid="stMetric"] {
        background: #fff; border: 1px solid #d7e0e9;
        border-radius: 4px; padding: .6rem .75rem;
    }
    div[data-testid="stMetricLabel"] { color: #50657c; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 4px; border-color: #9aafc5;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 1.2rem; border-bottom: 1px solid #cbd5e1; }
    .stTabs [data-baseweb="tab"] { color: #344b63; padding: .55rem .2rem; }
    .stTabs [aria-selected="true"] { color: #1769aa; }
    code, .stCode { color: #102f50 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_classifier() -> ModulationClassifier:
    if not MODEL_PATH.exists():
        train_and_save_model()
    return ModulationClassifier(MODEL_PATH)


def estimate_signal_quality(
    signal: np.ndarray,
    modulation: str,
    samples_per_symbol: int,
) -> tuple[float, float, float]:
    signal = np.asarray(signal, dtype=np.complex128).ravel()
    signal_power = float(np.mean(np.abs(signal) ** 2))
    chunks = [
        signal[start:start + samples_per_symbol]
        for start in range(0, signal.size - samples_per_symbol + 1, samples_per_symbol)
    ]
    residuals: list[np.ndarray] = []

    if modulation == 'BFSK':
        decisions = demodulate_signal(signal, modulation, samples_per_symbol)
        f0, f1 = bfsk_frequencies(samples_per_symbol)
        for index, chunk in enumerate(chunks[:decisions.size]):
            frequency = f1 if decisions[index] else f0
            n = np.arange(chunk.size, dtype=np.float64)
            omega = 2.0 * np.pi * frequency
            basis = np.column_stack((np.cos(omega * n), np.sin(omega * n)))
            coefficients, _, _, _ = np.linalg.lstsq(basis, np.real(chunk), rcond=None)
            residuals.append(np.real(chunk) - basis @ coefficients)
        residual_power = (
            float(np.mean(np.concatenate(residuals) ** 2)) if residuals else float('nan')
        )
        estimated_signal_power = max(signal_power - residual_power, 0.0)
    else:
        for chunk in chunks:
            residuals.append(chunk - np.mean(chunk))
        if residuals:
            residual_power = float(np.mean(np.abs(np.concatenate(residuals)) ** 2))
            residual_power *= samples_per_symbol / (samples_per_symbol - 1)
        else:
            residual_power = float('nan')
        estimated_signal_power = max(signal_power - residual_power, 0.0)

    if not np.isfinite(residual_power) or residual_power <= 0 or estimated_signal_power <= 0:
        snr_db = float('nan')
    else:
        snr_db = float(10.0 * np.log10(estimated_signal_power / residual_power))
    return estimated_signal_power, residual_power, snr_db


def analyze_received_signal(
    signal: np.ndarray,
    samples_per_symbol: int,
    sampling_frequency: float,
    source: str,
    bits: np.ndarray | None = None,
    reference_modulation: str | None = None,
    requested_snr: float | None = None,
    csv_bytes: bytes | None = None,
) -> None:
    st.session_state.pop('analysis_result', None)
    try:
        with st.spinner('Analyzing received signal…'):
            classifier = get_classifier()
            predicted, probabilities = classifier.predict_with_confidence(
                signal,
                samples_per_symbol=samples_per_symbol,
            )
            recovered = demodulate_signal(
                signal,
                predicted,
                samples_per_symbol=samples_per_symbol,
            )
            estimated_signal_power, noise_power, snr_estimate = estimate_signal_quality(
                signal,
                predicted,
                samples_per_symbol,
            )
        st.session_state['analysis_result'] = {
            'signal': np.asarray(signal, dtype=np.complex128),
            'samples_per_symbol': samples_per_symbol,
            'sampling_frequency': sampling_frequency,
            'source': source,
            'predicted': predicted,
            'probabilities': probabilities,
            'recovered': recovered,
            'estimated_signal_power': estimated_signal_power,
            'noise_power': noise_power,
            'snr_estimate': snr_estimate,
            'bits': bits,
            'reference_modulation': reference_modulation,
            'requested_snr': requested_snr,
            'csv_bytes': csv_bytes,
        }
    except (ValueError, OSError, KeyError) as exc:
        st.error(f'Signal analysis failed: {exc}')


def format_frequency(frequency: float) -> str:
    if frequency >= 1e6:
        return f'{frequency / 1e6:g} MHz'
    if frequency >= 1e3:
        return f'{frequency / 1e3:g} kHz'
    return f'{frequency:g} Hz'


def signal_information(analysis: dict) -> pd.DataFrame:
    signal = analysis['signal']
    if analysis['requested_snr'] is not None:
        snr_text = f'{analysis["requested_snr"]:g} dB (simulation setting)'
    elif np.isfinite(analysis['snr_estimate']):
        snr_text = f'~{analysis["snr_estimate"]:.2f} dB (estimated)'
    else:
        snr_text = 'Not available'
    return pd.DataFrame(
        {
            'Parameter': ('Samples', 'Sampling frequency', 'Signal power', 'SNR'),
            'Value': (
                f'{signal.size:,}',
                format_frequency(analysis['sampling_frequency']),
                f'{np.mean(np.abs(signal) ** 2):.6g}',
                snr_text,
            ),
        }
    )


def render_signal_plots(analysis: dict) -> None:
    signal = analysis['signal']
    sampling_frequency = analysis['sampling_frequency']
    stride = max(1, signal.size // 6000)
    samples = np.arange(signal.size)[::stride]
    time = samples / sampling_frequency
    frequency = np.fft.fftshift(np.fft.fftfreq(signal.size, d=1.0 / sampling_frequency))
    spectrum = np.fft.fftshift(np.fft.fft(signal))

    waveform_col, spectrum_col = st.columns(2)
    with waveform_col:
        st.markdown('<div class="section-title">Received signal</div>', unsafe_allow_html=True)
        fig, axis = plt.subplots(figsize=(7.2, 3.2), dpi=110)
        axis.plot(time, signal.real[::stride], color='#2475ad', linewidth=0.85, label='I / real')
        if np.any(np.abs(signal.imag) > 1e-12):
            axis.plot(time, signal.imag[::stride], color='#d07835', linewidth=0.75, alpha=0.8, label='Q / imaginary')
            axis.legend(loc='upper right', frameon=False, fontsize=8)
        axis.set_xlabel('Time (s)')
        axis.set_ylabel('Amplitude')
        axis.grid(True, color='#dce3ea', linewidth=0.6)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with spectrum_col:
        st.markdown('<div class="section-title">Frequency spectrum</div>', unsafe_allow_html=True)
        fig, axis = plt.subplots(figsize=(7.2, 3.2), dpi=110)
        axis.plot(frequency, np.abs(spectrum), color='#2475ad', linewidth=0.85)
        axis.set_xlabel('Frequency (Hz)')
        axis.set_ylabel('Magnitude')
        axis.grid(True, color='#dce3ea', linewidth=0.6)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)


def ideal_constellation(modulation: str) -> np.ndarray:
    if modulation == 'BPSK':
        return np.array([-1.0, 1.0], dtype=np.complex128)
    if modulation == 'QPSK':
        return np.exp(1j * np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]))
    if modulation == 'DPSK':
        return np.array([1.0, -1.0], dtype=np.complex128)
    if modulation == '16-QAM':
        levels = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
        return np.array([i + 1j * q for i in levels for q in levels])
    return np.array([], dtype=np.complex128)


def compare_simulation_bits(analysis: dict) -> tuple[int, int, float, float]:
    return compare_known_bits(
        analysis['bits'],
        analysis['recovered'],
        analysis['reference_modulation'],
    )


def render_analysis(analysis: dict | None) -> None:
    if analysis is None:
        st.info('Generate a reference signal or upload an unknown signal from the Signal Input panel.')
        return

    signal = analysis['signal']
    predicted = analysis['predicted']
    recovered = analysis['recovered']
    probabilities = analysis['probabilities']
    confidence = float(probabilities.get(predicted, 0.0))
    samples_per_symbol = analysis['samples_per_symbol']
    signal_power = float(np.mean(np.abs(signal) ** 2))

    st.markdown('<div class="section-title">Signal information</div>', unsafe_allow_html=True)
    st.dataframe(signal_information(analysis), hide_index=True, width='stretch')
    if analysis['csv_bytes'] is not None:
        filename = f'reference_signal_{analysis["reference_modulation"]}_{analysis["requested_snr"]:g}dB.csv'
        st.download_button(
            'Download signal CSV',
            data=analysis['csv_bytes'],
            file_name=filename,
            mime='text/csv',
        )
        st.caption('CSV columns: sample_index,I,Q. This exact file can be uploaded for signal analysis.')

    render_signal_plots(analysis)

    st.markdown('<div class="section-title">Modulation result</div>', unsafe_allow_html=True)
    result_col, probability_col = st.columns([1, 1.25])
    with result_col:
        st.metric('Detected modulation', predicted)
        st.metric('Confidence', f'{confidence:.1%}')
        if probabilities.get('BPSK', 0.0) + probabilities.get('DPSK', 0.0) > 0.1:
            st.caption(
                'Uncoded BPSK and DBPSK can be statistically indistinguishable without '
                'protocol structure such as a preamble.'
            )
    probability_table = pd.DataFrame(
        {
            'Modulation': CLASS_ORDER,
            'Probability': [float(probabilities.get(label, 0.0)) for label in CLASS_ORDER],
        }
    )
    with probability_col:
        st.dataframe(
            probability_table.style.format({'Probability': '{:.2%}'}),
            hide_index=True,
            width='stretch',
        )
        fig, axis = plt.subplots(figsize=(6.8, 2.2), dpi=110)
        axis.barh(
            probability_table['Modulation'],
            probability_table['Probability'] * 100,
            color=['#2475ad' if label == predicted else '#9aafc5' for label in CLASS_ORDER],
        )
        axis.set_xlabel('Probability (%)')
        axis.set_xlim(0, 100)
        axis.grid(axis='x', color='#dce3ea', linewidth=0.6)
        axis.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    st.markdown('<div class="section-title">Demodulation</div>', unsafe_allow_html=True)
    st.write(f'**Detected modulation:** {predicted}')
    st.write(f'**Demodulation method:** {DEMODULATION_METHODS[predicted]}')
    st.write('**Status:** Completed')

    method_table = pd.DataFrame(
        {
            'Detected Modulation': CLASS_ORDER,
            'Demodulation Method': [DEMODULATION_METHODS[label] for label in CLASS_ORDER],
        }
    )
    st.dataframe(
        method_table.style.apply(
            lambda column: (
                ['background-color: #dcecf8; font-weight: bold' if method == DEMODULATION_METHODS[predicted] else '' for method in column]
                if column.name == 'Demodulation Method'
                else [''] * len(column)
            ),
            axis=0,
        ),
        hide_index=True,
        width='stretch',
    )

    st.markdown('<div class="section-title">Recovered data</div>', unsafe_allow_html=True)
    st.caption(
        f'{len(recovered):,} recovered bits · '
        f'{(signal.size + samples_per_symbol - 1) // samples_per_symbol:,} received symbols · '
        f'{BITS_PER_SYMBOL[predicted]} bits/symbol'
    )
    bit_text = ''.join(recovered.astype(str).tolist())
    st.code(bit_text, language=None)
    st.download_button(
        'Download recovered data',
        data=bit_text + '\n',
        file_name='recovered_bits.txt',
        mime='text/plain',
    )

    st.markdown('<div class="section-title">Data recovery results</div>', unsafe_allow_html=True)
    if analysis['bits'] is None:
        st.info('BER: Not available. Original transmitted data was not provided.')
        recovery_rows = [
            ('Detected modulation', predicted),
            ('Demodulation method', DEMODULATION_METHODS[predicted]),
            ('Recovered bits', f'{len(recovered):,}'),
            ('Bit errors', 'Not available'),
            ('BER', 'Not available'),
            ('Bit accuracy', 'Not available'),
            ('Signal power', f'{signal_power:.6g}'),
            ('Estimated noise power', f'{analysis["noise_power"]:.6g}' if np.isfinite(analysis['noise_power']) else 'Not available'),
        ]
    else:
        errors, compared, ber, accuracy = compare_simulation_bits(analysis)
        recovery_rows = [
            ('Detected modulation', predicted),
            ('Demodulation method', DEMODULATION_METHODS[predicted]),
            ('Recovered bits', f'{len(recovered):,}'),
            ('Compared bits', f'{compared:,}'),
            ('Bit errors', f'{errors:,}'),
            ('BER', f'{ber:.6f}'),
            ('Bit accuracy', f'{accuracy:.2%}'),
            ('Signal power', f'{signal_power:.6g}'),
            ('Estimated noise power', f'{analysis["noise_power"]:.6g}' if np.isfinite(analysis['noise_power']) else 'Not available'),
        ]
        if analysis['reference_modulation'] != predicted:
            st.warning(
                'The reference signal was classified differently. The demodulator was still '
                'selected from the detected modulation, not the reference setting.'
            )
    st.dataframe(
        pd.DataFrame(recovery_rows, columns=['Parameter', 'Result']),
        hide_index=True,
        width='stretch',
    )
    st.session_state['last_recovery_results'] = recovery_rows

    if predicted in ('QPSK', '16-QAM'):
        st.markdown('<div class="section-title">Constellation</div>', unsafe_allow_html=True)
        stride = max(1, signal.size // 3000)
        fig, axis = plt.subplots(figsize=(5.8, 4.0), dpi=110)
        axis.scatter(signal.real[::stride], signal.imag[::stride], s=9, alpha=0.35, color='#2475ad')
        reference = ideal_constellation(predicted)
        axis.scatter(reference.real, reference.imag, marker='x', s=45, color='#b24a3a', label='Ideal symbol locations')
        axis.set_xlabel('In-phase (I)')
        axis.set_ylabel('Quadrature (Q)')
        axis.grid(True, color='#dce3ea', linewidth=0.6)
        axis.set_aspect('equal', adjustable='datalim')
        axis.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)


def run_snr_evaluation(classifier: ModulationClassifier, repetitions: int) -> pd.DataFrame:
    rng = np.random.default_rng()
    rows = []
    for modulation in CLASS_ORDER:
        for snr_db in SNR_LEVELS:
            for _ in range(repetitions):
                bits = generate_random_bits(256, rng=rng)
                signal = generate_modulated_signal(
                    modulation,
                    bits=bits,
                    snr_db=float(snr_db),
                    rng=rng,
                    samples_per_symbol=EVALUATION_SAMPLES_PER_SYMBOL,
                )
                predicted, _ = classifier.predict_with_confidence(
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
                _, compared, ber, accuracy = compare_known_bits(bits, automatic_bits, modulation)
                _, oracle_compared, oracle_ber, oracle_accuracy = compare_known_bits(
                    bits,
                    oracle_bits,
                    modulation,
                )
                rows.append(
                    {
                        'snr_db': snr_db,
                        'actual_modulation': modulation,
                        'predicted_modulation': predicted,
                        'classification_correct': predicted == modulation,
                        'compared_bits': compared,
                        'ber': ber,
                        'demodulation_accuracy': accuracy,
                        'oracle_compared_bits': oracle_compared,
                        'oracle_ber': oracle_ber,
                        'oracle_demodulation_accuracy': oracle_accuracy,
                        'samples_per_symbol': EVALUATION_SAMPLES_PER_SYMBOL,
                    }
                )
    return pd.DataFrame(rows)


def render_confusion_matrix(matrix: np.ndarray) -> None:
    fig, axis = plt.subplots(figsize=(6.0, 4.6), dpi=110)
    image = axis.imshow(matrix, cmap='Blues')
    axis.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=35, ha='right')
    axis.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    axis.set_xlabel('Predicted modulation')
    axis.set_ylabel('True modulation')
    axis.set_title('Confusion matrix')
    threshold = float(matrix.max()) / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f'{int(matrix[row, column])}',
                ha='center',
                va='center',
                color='white' if matrix[row, column] > threshold else '#172b45',
                fontsize=9,
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def plot_snr_results(results: pd.DataFrame) -> None:
    classification = (
        results.groupby(['snr_db', 'actual_modulation'])['classification_correct']
        .mean()
        .unstack('actual_modulation')
        .reindex(columns=CLASS_ORDER)
    )
    demodulation = (
        results.groupby(['snr_db', 'actual_modulation'])['demodulation_accuracy']
        .mean()
        .unstack('actual_modulation')
        .reindex(columns=CLASS_ORDER)
    )
    ber = (
        results.groupby(['snr_db', 'actual_modulation'])['ber']
        .mean()
        .unstack('actual_modulation')
        .reindex(columns=CLASS_ORDER)
    )
    accuracy_plot, demod_plot, ber_plot = st.tabs(
        ['Classification accuracy vs SNR', 'Demodulation accuracy vs SNR', 'BER vs SNR']
    )
    for container, data, title, ylabel in (
        (accuracy_plot, classification, 'Classification accuracy vs SNR', 'Accuracy (%)'),
        (demod_plot, demodulation, 'Demodulation accuracy vs SNR', 'Accuracy (%)'),
    ):
        with container:
            fig, axis = plt.subplots(figsize=(8, 4.0), dpi=110)
            for label in CLASS_ORDER:
                axis.plot(data.index, data[label] * 100, marker='o', markersize=3, label=label)
            axis.set_title(title)
            axis.set_xlabel('SNR (dB)')
            axis.set_ylabel(ylabel)
            axis.set_ylim(0, 102)
            axis.grid(True, color='#dce3ea', linewidth=0.6)
            axis.legend(frameon=False, ncol=3, fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
    with ber_plot:
        fig, axis = plt.subplots(figsize=(8, 4.0), dpi=110)
        for label in CLASS_ORDER:
            positive = ber[label].where(ber[label] > 0)
            axis.semilogy(ber.index, positive, marker='o', markersize=3, label=label)
            zero_count = int((ber[label] == 0).sum())
            if zero_count:
                axis.plot([], [], ' ', label=f'{label}: {zero_count} zero-BER points omitted')
        axis.set_title('BER vs SNR')
        axis.set_xlabel('SNR (dB)')
        axis.set_ylabel('Bit error rate (log scale)')
        axis.grid(True, which='both', color='#dce3ea', linewidth=0.6)
        axis.legend(frameon=False, ncol=2, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.caption('Measured zero-BER points are omitted from the logarithmic axis because log(0) is undefined.')


def render_performance(classifier: ModulationClassifier) -> None:
    st.markdown('<div class="section-title">Hold-out classification results</div>', unsafe_allow_html=True)
    bundle = classifier.bundle
    report = bundle.get('classification_report', {})
    metrics = [
        ('Classification accuracy', float(bundle.get('accuracy', report.get('accuracy', float('nan'))))),
        ('Precision', float(report.get('macro avg', {}).get('precision', float('nan')))),
        ('Recall', float(report.get('macro avg', {}).get('recall', float('nan')))),
        ('F1 score', float(report.get('macro avg', {}).get('f1-score', float('nan')))),
    ]
    metric_columns = st.columns(4)
    for column, (label, value) in zip(metric_columns, metrics):
        column.metric(label, f'{value:.2%}' if np.isfinite(value) else 'Not available')

    labels = bundle.get('labels', list(CLASS_ORDER))
    stored_matrix = np.asarray(bundle.get('confusion_matrix', []))
    if stored_matrix.shape == (len(labels), len(labels)):
        indices = [labels.index(label) for label in CLASS_ORDER]
        matrix = stored_matrix[np.ix_(indices, indices)]
        st.markdown('<div class="section-title">Confusion matrix</div>', unsafe_allow_html=True)
        render_confusion_matrix(matrix)

    st.markdown('<div class="section-title">SNR performance</div>', unsafe_allow_html=True)
    st.caption(
        'Fresh independent signals are used. The reference modulation is used only after '
        'classification to score results; the predicted modulation selects the automatic demodulator.'
    )
    repetitions = st.selectbox('Signals per modulation and SNR', (1, 2, 5, 10), index=1)
    if st.button('Run SNR analysis', type='primary'):
        try:
            with st.spinner('Generating signals and measuring system performance…'):
                results = run_snr_evaluation(classifier, repetitions)
                METRICS_DIR.mkdir(parents=True, exist_ok=True)
                results.to_csv(EVALUATION_PATH, index=False)
                st.session_state['snr_evaluation'] = results
        except (ValueError, OSError) as exc:
            st.error(f'SNR evaluation failed: {exc}')

    results = st.session_state.get('snr_evaluation')
    if results is None and EVALUATION_PATH.exists():
        try:
            results = pd.read_csv(EVALUATION_PATH)
        except (OSError, pd.errors.ParserError) as exc:
            st.error(f'Could not load saved SNR measurements: {exc}')
    if results is None:
        st.info('No SNR measurements are available yet. Run the analysis above to create measured results.')
        return

    required = {
        'snr_db',
        'actual_modulation',
        'predicted_modulation',
        'classification_correct',
        'demodulation_accuracy',
        'ber',
    }
    if not required.issubset(results.columns):
        st.warning('Saved measurements use an older format. Run a new SNR analysis.')
        return

    if 'oracle_ber' not in results:
        results['oracle_ber'] = np.nan
        results['oracle_demodulation_accuracy'] = np.nan
    if 'oracle_demodulation_accuracy' not in results:
        results['oracle_demodulation_accuracy'] = np.nan

    st.markdown('<div class="section-title">Classification confusion by SNR</div>', unsafe_allow_html=True)
    selected_snr = st.select_slider(
        'SNR (dB)',
        options=list(SNR_LEVELS),
        value=15,
        key='performance_confusion_snr',
    )
    selected = results[results['snr_db'] == selected_snr]
    confusion = pd.crosstab(
        pd.Categorical(selected['actual_modulation'], categories=CLASS_ORDER, ordered=True),
        pd.Categorical(selected['predicted_modulation'], categories=CLASS_ORDER, ordered=True),
        dropna=False,
    ).to_numpy()
    render_confusion_matrix(confusion)

    plot_snr_results(results)
    st.download_button(
        'Download measured results',
        data=results.to_csv(index=False),
        file_name='modulation_snr_results.csv',
        mime='text/csv',
    )


def render_about() -> None:
    st.markdown('<div class="section-title">Project overview</div>', unsafe_allow_html=True)
    st.write(
        'To identify the modulation scheme of a received digital communication signal and '
        'recover the transmitted data using the corresponding demodulation technique.'
    )

    st.markdown('<div class="section-title">Supported modulation schemes</div>', unsafe_allow_html=True)
    st.write(', '.join(CLASS_ORDER))

    st.markdown('<div class="section-title">Signal processing stages</div>', unsafe_allow_html=True)
    st.write(
        'Signal generation · AWGN channel · Feature extraction · Modulation classification · '
        'Automatic demodulator selection · Data recovery'
    )

    st.markdown('<div class="section-title">Feature extraction</div>', unsafe_allow_html=True)
    st.write(
        'The received waveform is summarized using amplitude and I/Q statistics, phase and '
        'instantaneous-frequency measurements, spectral descriptors, and signal-distribution statistics.'
    )

    st.markdown('<div class="section-title">Classification and system architecture</div>', unsafe_allow_html=True)
    st.write(
        'A trained Random Forest classifier operates on standardized signal features. '
        'For an unknown input, only received samples enter classification. The predicted class '
        'selects the matching demodulator. In simulation, reference bits and the selected generator '
        'scheme are retained solely for post-recovery scoring.'
    )
    st.code(
        'Received samples → Features → Modulation result → Demodulator → Recovered data',
        language=None,
    )
    st.caption(
        'Uploaded data is expected to be sampled baseband signal. Carrier recovery and automatic '
        'symbol-timing synchronization are not implemented; provide the sampling frequency and samples per symbol.'
    )


st.title('Automatic Modulation Recognition')
st.markdown('<div class="subtitle">Signal Analysis and Data Recovery</div>', unsafe_allow_html=True)
st.markdown('<hr style="border:0;border-top:1px solid #cbd5e1;margin:.2rem 0 1rem">', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('### SIGNAL INPUT')
    input_mode = st.radio(
        'Input type',
        ('Reference Signal Generation', 'Upload Signal'),
        label_visibility='collapsed',
    )
    previous_input_mode = st.session_state.get('previous_input_mode')
    if previous_input_mode is not None and previous_input_mode != input_mode:
        st.session_state.pop('analysis_result', None)
    st.session_state['previous_input_mode'] = input_mode

    if input_mode == 'Reference Signal Generation':
        st.caption('Reference Signal Generation · simulation and testing only.')
        with st.form('generate_signal_form'):
            bit_length = st.selectbox('Number of bits', BIT_LENGTH_OPTIONS)
            samples_per_symbol = st.selectbox(
                'Samples per symbol',
                SAMPLES_PER_SYMBOL_OPTIONS,
                index=2,
            )
            simulation_snr = st.slider(
                'SNR (dB)',
                min_value=-10,
                max_value=30,
                value=15,
                step=5,
            )
            reference_modulation = st.selectbox('Reference modulation', CLASS_ORDER)
            generate_clicked = st.form_submit_button('Generate Signal', type='primary')
        if generate_clicked:
            st.session_state.pop('analysis_result', None)
            try:
                reference_bits = generate_random_bits(bit_length)
                received = generate_modulated_signal(
                    reference_modulation,
                    bits=reference_bits,
                    snr_db=float(simulation_snr),
                    samples_per_symbol=samples_per_symbol,
                )
                generated_csv = signal_to_csv_bytes(received)
                loaded = load_signal_csv(BytesIO(generated_csv))
                sampling_frequency = 10000.0
                generated_digest = hashlib.sha256(generated_csv).hexdigest()
                st.session_state['generated_csv_metadata'] = {
                    generated_digest: {
                        'bits': reference_bits.copy(),
                        'reference_modulation': reference_modulation,
                        'requested_snr': float(simulation_snr),
                        'samples_per_symbol': samples_per_symbol,
                        'sampling_frequency': sampling_frequency,
                    }
                }
                analyze_received_signal(
                    loaded.signal,
                    samples_per_symbol,
                    sampling_frequency,
                    'simulation',
                    bits=reference_bits,
                    reference_modulation=reference_modulation,
                    requested_snr=float(simulation_snr),
                    csv_bytes=generated_csv,
                )
            except ValueError as exc:
                st.error(f'Could not generate the reference signal: {exc}')
    else:
        st.caption('Required CSV columns: sample_index,I,Q.')
        with st.form('upload_signal_form'):
            uploaded_file = st.file_uploader('Upload CSV', type=['csv'])
            upload_sps = st.selectbox(
                'Samples per symbol',
                SAMPLES_PER_SYMBOL_OPTIONS,
                index=2,
                help='Symbol timing parameter; it does not specify the modulation.',
            )
            upload_fs = st.number_input(
                'Sampling frequency (Hz)',
                min_value=0.000001,
                value=10000.0,
                step=1000.0,
                format='%.3f',
            )
            analyze_clicked = st.form_submit_button('Analyze Signal', type='primary')
        if analyze_clicked:
            st.session_state.pop('analysis_result', None)
            if uploaded_file is None:
                st.error('Choose a CSV file before analysis.')
            else:
                try:
                    uploaded_bytes = uploaded_file.getvalue()
                    loaded = load_signal_csv(uploaded_bytes)
                    uploaded_digest = hashlib.sha256(uploaded_bytes).hexdigest()
                    reference_metadata = st.session_state.get(
                        'generated_csv_metadata',
                        {},
                    ).get(uploaded_digest)
                    if reference_metadata is not None:
                        st.info(
                            'This is the byte-identical CSV generated in this session. '
                            'Using its original symbol timing and sampling frequency for analysis.'
                        )
                        upload_sps = reference_metadata['samples_per_symbol']
                        upload_fs = reference_metadata['sampling_frequency']
                    analyze_received_signal(
                        loaded.signal,
                        upload_sps,
                        upload_fs,
                        'upload',
                        bits=reference_metadata['bits'] if reference_metadata else None,
                        reference_modulation=(
                            reference_metadata['reference_modulation']
                            if reference_metadata
                            else None
                        ),
                        requested_snr=(
                            reference_metadata['requested_snr']
                            if reference_metadata
                            else None
                        ),
                    )
                except ValueError as exc:
                    st.error(str(exc))

analysis_tab, performance_tab, about_tab = st.tabs(['ANALYSIS', 'PERFORMANCE', 'ABOUT'])
analysis_result = st.session_state.get('analysis_result')
with analysis_tab:
    render_analysis(analysis_result)
with performance_tab:
    try:
        render_performance(get_classifier())
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f'Performance results are unavailable: {exc}')
with about_tab:
    render_about()
