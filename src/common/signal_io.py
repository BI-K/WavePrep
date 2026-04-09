"""Readers and writers for multiple waveform output formats.

Writers use WFDB for native WFDB and MATLAB output, pyedflib for EDF,
soundfile for WAV, and pandas for CSV.

Readers normalise any supported format back into (rows, channel_names)
tuples for downstream consumption (e.g. Croissant manifest building).

The WFDB MATLAB conversion operates on an on-disk WFDB record, so the
workflow for MATLAB is:
  1. Write a temporary WFDB record via wfdb.wrsamp()
  2. Convert to the target format
  3. Remove the temporary WFDB files
"""

import os
import logging
import math
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import wfdb
from wfdb.io.convert import wfdb_to_mat

logger = logging.getLogger(__name__)

# Maps format name → file extension used for output files
FORMAT_EXTENSIONS: Dict[str, str] = {
    'csv': '.csv',
    'edf': '.edf',
    'matlab': '.mat',
    'wav': '.wav',
    'wfdb': '.dat',  # WFDB produces .dat + .hea
}


def _write_csv(filepath: Path, data: np.ndarray, channel_names: List[str],
               fs: float) -> None:
    """Write signal data as CSV."""
    df = pd.DataFrame(data, columns=channel_names)
    df.to_csv(filepath, index=False)


def _write_wfdb(filepath: Path, data: np.ndarray, channel_names: List[str],
                fs: float) -> None:
    """Write signal data as a WFDB record (.dat + .hea)."""
    record_name = filepath.stem
    write_dir = str(filepath.parent)
    units = ['mV'] * len(channel_names)
    wfdb.wrsamp(
        record_name,
        fs=fs,
        units=units,
        sig_name=channel_names,
        p_signal=data.astype(np.float64),
        fmt=['16'] * len(channel_names),
        write_dir=write_dir,
    )


def _write_edf(filepath: Path, data: np.ndarray, channel_names: List[str],
               fs: float) -> None:
    """Write signal data as EDF using pyedflib.

    WFDB's `wfdb_to_edf` conversion truncates scientific notation in fixed-width
    EDF physical min/max fields for some short windows, producing unreadable
    files. Writing EDF directly avoids that corruption and preserves the actual
    window length instead of forcing 10-second blocks.
    """
    import pyedflib

    signal_matrix = np.asarray(data, dtype=np.float64)
    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix[:, np.newaxis]

    signals = [signal_matrix[:, index] for index in range(signal_matrix.shape[1])]
    signal_headers = []

    for channel_name, signal in zip(channel_names, signals):
        physical_min = float(np.min(signal))
        physical_max = float(np.max(signal))
        if physical_min == physical_max:
            padding = max(abs(physical_min) * 1e-3, 1e-6)
        else:
            padding = max((physical_max - physical_min) * 1e-3, 1e-6)

        signal_headers.append({
            'label': channel_name[:16],
            'dimension': 'mV',
            'sample_frequency': fs,
            'physical_min': physical_min - padding,
            'physical_max': physical_max + padding,
            'digital_min': -32768,
            'digital_max': 32767,
            'transducer': '',
            'prefilter': '',
        })

    total_samples = int(signal_matrix.shape[0])
    min_samples_per_record = max(1, int(math.ceil(fs * 0.001 - 1e-9)))
    max_samples_per_record = max(
        min_samples_per_record,
        min(total_samples, int(math.floor(fs * 60 + 1e-9))),
    )

    samples_per_record = None
    for candidate in range(max_samples_per_record, min_samples_per_record - 1, -1):
        if total_samples % candidate == 0:
            samples_per_record = candidate
            break

    if samples_per_record is None:
        samples_per_record = max_samples_per_record

    record_duration = samples_per_record / float(fs)

    with pyedflib.EdfWriter(
        str(filepath),
        n_channels=len(signals),
        file_type=pyedflib.FILETYPE_EDF,
    ) as writer:
        writer.setSignalHeaders(signal_headers)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message='Forcing a specific record_duration might alter calculated sample_frequencies when reading the file',
            )
            writer.setDatarecordDuration(record_duration)
        writer.writeSamples(signals, digital=False)


def _write_matlab(filepath: Path, data: np.ndarray, channel_names: List[str],
                  fs: float) -> None:
    """Write signal data as MATLAB .mat via WFDB intermediate."""
    record_name = filepath.stem
    write_dir = str(filepath.parent)

    units = ['mV'] * len(channel_names)
    wfdb.wrsamp(
        record_name,
        fs=fs,
        units=units,
        sig_name=channel_names,
        p_signal=data.astype(np.float64),
        fmt=['16'] * len(channel_names),
        write_dir=write_dir,
    )

    prev_cwd = os.getcwd()
    try:
        os.chdir(write_dir)
        wfdb_to_mat(record_name)
    finally:
        os.chdir(prev_cwd)

    # wfdb_to_mat produces {sanitized_name}m.mat and {sanitized_name}m.hea
    # where sanitized_name replaces hyphens with underscores
    mat_name = record_name.replace('-', '_')
    mat_src = Path(write_dir) / f"{mat_name}m.mat"
    mat_dst = Path(write_dir) / f"{record_name}.mat"
    if mat_src.exists() and mat_src != mat_dst:
        mat_dst.unlink(missing_ok=True)
        mat_src.rename(mat_dst)

    # Clean up temporary files
    for pattern in (f"{record_name}.dat", f"{record_name}.hea",
                    f"{mat_name}m.hea"):
        tmp = Path(write_dir) / pattern
        if tmp.exists():
            tmp.unlink()


def _write_wav(filepath: Path, data: np.ndarray, channel_names: List[str],
               fs: float) -> None:
    """Write signal data as WAV using soundfile (DOUBLE subtype for precision)."""
    import soundfile as sf

    output_path = filepath.with_suffix('.wav')
    sf.write(str(output_path), data.astype(np.float64), int(fs), subtype='DOUBLE')

    # Also write a sidecar JSON with channel names (WAV has no channel name metadata)
    sidecar = output_path.with_suffix('.wav.json')
    import json
    with open(sidecar, 'w') as f:
        json.dump({'channel_names': channel_names, 'fs': fs}, f)


# Registry of writers keyed by format name
_WRITERS = {
    'csv': _write_csv,
    'edf': _write_edf,
    'matlab': _write_matlab,
    'wav': _write_wav,
    'wfdb': _write_wfdb,
}


def get_signal_writer(fmt: str):
    """Return a writer function for the given format name.

    Parameters
    ----------
    fmt : str
        One of: csv, edf, matlab, wav, wfdb

    Returns
    -------
    Callable[[Path, np.ndarray, List[str], float], None]
    """
    fmt = fmt.lower()
    writer = _WRITERS.get(fmt)
    if writer is None:
        raise ValueError(
            f"Unsupported save format '{fmt}'. "
            f"Supported: {sorted(_WRITERS)}"
        )
    return writer


def get_file_extension(fmt: str) -> str:
    """Return the file extension (with dot) for the given format."""
    return FORMAT_EXTENSIONS.get(fmt.lower(), '.csv')


def read_signal_data(
    path: Path,
    save_format: str,
) -> tuple[list[list[float]], list[str]]:
    """Read a sample file and return (rows_as_nested_lists, channel_names).

    Unified reader for all supported formats. Returns data as a list of rows
    (each row is a list of per-channel floats) plus ordered channel names.
    """
    fmt = save_format.lower()

    if fmt == 'csv':
        df = pd.read_csv(path)
        return df.values.tolist(), list(df.columns)

    if fmt == 'edf':
        import pyedflib
        reader = pyedflib.EdfReader(str(path))
        try:
            n = reader.signals_in_file
            signals = [reader.readSignal(i) for i in range(n)]
            channels = [reader.getLabel(i).strip() for i in range(n)]
            return np.column_stack(signals).tolist(), channels
        finally:
            reader.close()

    if fmt == 'matlab':
        from scipy.io import loadmat
        contents = loadmat(str(path))
        arrays = {
            k: v for k, v in contents.items()
            if not k.startswith('__') and isinstance(v, np.ndarray)
        }
        _, value = max(arrays.items(), key=lambda item: item[1].size)
        arr = np.asarray(value)
        if arr.ndim == 2:
            arr = arr.T  # MATLAB stores (channels, samples)
        channels = [f"ch_{i}" for i in range(arr.shape[-1] if arr.ndim > 1 else 1)]
        return arr.tolist(), channels

    if fmt == 'wav':
        import json
        import soundfile as sf
        data, _ = sf.read(str(path), always_2d=True)
        sidecar = path.with_suffix('.wav.json')
        if sidecar.exists():
            meta = json.loads(sidecar.read_text(encoding='utf-8'))
            channels = meta.get('channel_names', [])
        else:
            channels = [f"ch_{i}" for i in range(data.shape[1])]
        return data.tolist(), channels

    if fmt == 'wfdb':
        record = wfdb.rdrecord(str(path.with_suffix('')))
        data = record.p_signal if record.p_signal is not None else record.d_signal
        return np.asarray(data).tolist(), list(record.sig_name)

    raise ValueError(f"Unsupported format for reading: {fmt}")
