#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.io import savemat

from pingdata.data import PingMetadata, read_ping_metadata, read_sonar_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def construct_output_dir(input_file: Path, output_dir: Path | None = None) -> Path:
    """Get the save path for the converted data."""
    if not output_dir:
        return input_file.parent / input_file.stem
    return output_dir / input_file.stem


def convert_data(
    input_file: Path, temperature: float = 10.0, sound_speed: float = 1500.0
) -> tuple[PingMetadata, dict[dict]]:
    ping_metadata = read_ping_metadata(input_file, temperature, sound_speed)
    return ping_metadata, read_sonar_data(ping_metadata)


def save_hdf5(beam_name: str, beam_data: dict, output_dir: Path) -> None:
    """Save beam data to a .hdf5 file."""
    import h5py

    with h5py.File(output_dir / f"{beam_name}.hdf5", "w") as f:
        for key, value in beam_data.items():
            f.create_dataset(key, data=value)


def save_mat(beam_name: str, beam_data: dict, output_dir: Path) -> None:
    """Save beam data to a .mat file."""
    savemat(
        output_dir / f"{beam_name}.mat",
        {
            "data": beam_data,
        },
    )


def save_npy(beam_name: str, beam_data: dict, output_dir: Path) -> None:
    """Save beam data to a .npy file."""
    np.save(output_dir / f"{beam_name}.npy", beam_data)


def save_function_factory(file_format: str):
    """Factory function to save data in different formats."""
    if file_format == "hdf5":
        return save_hdf5
    if file_format == "mat":
        return save_mat
    if file_format == "npy":
        return save_npy
    raise ValueError(f"Unsupported file format: {file_format}")


def save_data(
    metadata: PingMetadata, data: dict, output_dir: Path, file_format: str
) -> None:
    save_func = save_function_factory(file_format)
    for beam, (beam_name, beam_data) in zip(metadata.beams, data.items()):
        beam.metadata.to_csv(output_dir / f"{beam_name}.csv", index=False)
        save_func(beam_name, beam_data, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="PingData CLI")
    parser.add_argument("input", type=Path, help="Path to input file.")
    parser.add_argument("-o", "--outdir", type=Path, help="Output directory.")
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="mat",
        choices=["hdf5", "mat", "npy"],
        help="Output format.",
    )
    parser.add_argument(
        "-t",
        "--temp",
        type=float,
        default=10.0,
        help="Water temperature, C (default: 10.0).",
    )
    parser.add_argument(
        "-s",
        "--sound-speed",
        type=float,
        default=1500.0,
        help="Sound speed, m/s (default: 1500.0).",
    )
    args = parser.parse_args()

    logging.info(f"Reading {args.input.resolve()}.")
    metadata, data = convert_data(args.input, args.temp, args.sound_speed)
    output_dir = construct_output_dir(args.input, args.outdir)

    logging.info(
        f"Saving data to {output_dir.resolve()} in `{args.format.upper()}` format."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_data(metadata, data, output_dir, args.format)

    logging.info("Conversion complete.")


if __name__ == "__main__":
    main()
