# Automatic Modulation Recognition

Signal Analysis and Data Recovery

## Supported modulations
- BPSK
- BFSK
- QPSK
- DPSK
- 16-QAM

## Key behaviors
- AWGN channel simulation with configurable SNR levels
- Balanced training dataset generation with stratified splits
- Training examples span 4, 8, 16, and 32 samples per symbol
- Feature extraction for time, frequency, phase, and instantaneous-frequency statistics
- Automatic demodulation based on the predicted modulation
- End-to-end evaluation across SNR values
- Streamlit dashboard for unknown-signal CSV upload, simulation-only test generation,
  automatic demodulator selection, recovered-bit export, and measured SNR evaluation
- Three dashboard sections: Analysis, Performance, and About
- Generated signal downloads and signal uploads share the strict `sample_index,I,Q` CSV format

The upload workflow does not request a modulation label. It accepts CSV files with exactly
the columns `sample_index,I,Q`; the same parser also reads generated CSVs before they enter
the analysis pipeline. For unknown uploads, the user provides samples per symbol and sampling
frequency; these are signal timing parameters, not modulation choices. A byte-identical CSV
generated in the current session can reuse its in-memory reference bits for BER scoring, without
adding labels or metadata to the file. Other uploaded signals do not display BER without known bits.
Upload samples should be baseband data using this project's modulation conventions; carrier
recovery and automatic symbol-timing synchronization are not implemented.

The dashboard's classification probabilities are model scores, not calibrated guarantees.
In particular, equiprobable uncoded BPSK and DBPSK can produce statistically indistinguishable
symbol sequences; a preamble, coding structure, or other protocol-level information may be
required to reliably distinguish them.

## Quick start
1. Install dependencies:
   `pip install -r requirements.txt`
2. Train the model:
   `python train.py`
3. Run the Streamlit dashboard:
   `streamlit run app.py`
4. Evaluate the full pipeline:
   `python test.py`

The command-line evaluation generates fresh independent trials and reports both
oracle demodulation (true label used only after classification) and full automatic
demodulation (predicted label selects the demodulator).
