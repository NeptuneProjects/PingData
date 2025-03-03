#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from scipy.io import savemat

from pingdata.data import PingMetadata, read_ping_metadata, read_sonar_data


def construct_output_dir(input_file: Path, output_dir: Path | None = None) -> Path:
    """Get the save path for the converted data."""
    if not output_dir:
        return input_file.stem
    return output_dir / input_file.stem


def convert_data(
    input_file: Path, temperature: float = 10.0, sound_speed: float = 1500.0
) -> tuple:
    ping_metadata = read_ping_metadata(input_file, temperature, sound_speed)
    return ping_metadata, read_sonar_data(ping_metadata)


def save_data(metadata: PingMetadata, data: dict, output_dir: Path):
    for beam, (beam_name, beam_data) in zip(metadata.beams, data.items()):
        beam.metadata.to_csv(output_dir / f"{beam_name}.csv", index=False)
        savemat(
            output_dir / f"{beam_name}.mat",
            {
                "data": beam_data,
            },
        )


def main():
    parser = argparse.ArgumentParser(description="PingData CLI")
    parser.add_argument("input", type=Path, help="Path to input file.")
    parser.add_argument("-o", "--outdir", type=Path, help="Output directory.")
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

    metadata, data = convert_data(args.input, args.temp, args.sound_speed)
    output_dir = construct_output_dir(args.input, args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_data(metadata, data, output_dir)


if __name__ == "__main__":
    main()
