"""Writers for multiple waveform output formats.

Uses the wfdb library for WFDB, EDF, and MATLAB formats.
Uses soundfile for WAV format.
CSV uses pandas as before.

The wfdb conversion functions (wfdb_to_edf, wfdb_to_mat) operate on
on-disk WFDB records, so the workflow for EDF/MAT is:
  1. Write a temporary WFDB record via wfdb.wrsamp()
  2. Convert to the target format
  3. Remove the temporary WFDB files
"""

import os
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import wfdb
from wfdb.io.convert import wfdb_to_edf, wfdb_to_mat

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
    """Write signal data as EDF via WFDB intermediate."""
    record_name = filepath.stem
    write_dir = str(filepath.parent)

    # Write temporary WFDB record
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

    # Convert WFDB → EDF (wfdb_to_edf operates in cwd)
    prev_cwd = os.getcwd()
    try:
        os.chdir(write_dir)
        edf_filename = f"{record_name}.edf"
        wfdb_to_edf(record_name, output_filename=edf_filename)
    finally:
        os.chdir(prev_cwd)

    # Clean up temporary WFDB files
    for ext in ('.dat', '.hea'):
        tmp = Path(write_dir) / f"{record_name}{ext}"
        if tmp.exists():
            tmp.unlink()


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
