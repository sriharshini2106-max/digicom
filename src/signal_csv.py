from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import numpy as np
import pandas as pd

CSV_COLUMNS = ('sample_index', 'I', 'Q')
INVALID_FORMAT_MESSAGE = (
    'Invalid signal file format.\n\n'
    'Expected CSV columns:\n\n'
    'sample_index, I, Q'
)


@dataclass(frozen=True)
class LoadedSignalCsv:
    signal: np.ndarray
    sample_index: np.ndarray


def signal_to_csv_bytes(signal: np.ndarray) -> bytes:
    signal = np.asarray(signal, dtype=np.complex128).ravel()
    if signal.size == 0:
        raise ValueError('Cannot create a signal CSV from an empty signal.')
    if not np.isfinite(signal.real).all() or not np.isfinite(signal.imag).all():
        raise ValueError('Signal samples must be finite real or complex values.')

    dataframe = pd.DataFrame(
        {
            'sample_index': np.arange(signal.size, dtype=np.int64),
            'I': signal.real,
            'Q': signal.imag,
        },
        columns=CSV_COLUMNS,
    )
    return dataframe.to_csv(index=False, float_format='%.17g', lineterminator='\n').encode('utf-8')


def load_signal_csv(file: bytes | bytearray | BinaryIO) -> LoadedSignalCsv:
    source = BytesIO(file) if isinstance(file, (bytes, bytearray)) else file
    try:
        dataframe = pd.read_csv(source)
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(INVALID_FORMAT_MESSAGE) from exc

    if tuple(dataframe.columns) != CSV_COLUMNS:
        raise ValueError(INVALID_FORMAT_MESSAGE)
    if dataframe.empty or dataframe.isna().any().any():
        raise ValueError('Signal CSV must contain at least one complete row with no missing values.')

    try:
        sample_index_values = pd.to_numeric(dataframe['sample_index'], errors='raise').to_numpy(dtype=np.float64)
        i_values = pd.to_numeric(dataframe['I'], errors='raise').to_numpy(dtype=np.float64)
        q_values = pd.to_numeric(dataframe['Q'], errors='raise').to_numpy(dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('sample_index, I, and Q must contain numeric values.') from exc

    if (
        not np.isfinite(sample_index_values).all()
        or not np.isfinite(i_values).all()
        or not np.isfinite(q_values).all()
    ):
        raise ValueError('sample_index, I, and Q must contain finite numeric values.')
    if not np.equal(sample_index_values, np.floor(sample_index_values)).all():
        raise ValueError('sample_index values must be integers.')
    expected_index = np.arange(sample_index_values.size, dtype=np.int64)
    if not np.array_equal(sample_index_values, expected_index):
        raise ValueError('sample_index must start at 0 and increase by 1 for each row.')
    sample_index = expected_index

    signal = i_values + 1j * q_values
    if signal.size < 32:
        raise ValueError('At least 32 signal samples are required for analysis.')
    if float(np.mean(np.abs(signal) ** 2)) <= 1e-15:
        raise ValueError('The uploaded signal has zero power.')
    return LoadedSignalCsv(signal=signal, sample_index=sample_index)
