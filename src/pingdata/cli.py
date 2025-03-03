#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path

import pingdata.data as pdata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def instantiate_parser() -> argparse.ArgumentParser:
    """Instantiate the argument parser.

    Returns:
        Argument parser object.
    """
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
    return parser


def main() -> None:
    parser = instantiate_parser()
    args = parser.parse_args()

    logging.info(f"Reading {args.input.resolve()}.")
    metadata, data = pdata.convert_data(args.input, args.temp, args.sound_speed)
    output_dir = pdata.construct_output_dir(args.input, args.outdir)

    logging.info(
        f"Saving data to {output_dir.resolve()} in `{args.format.upper()}` format."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pdata.save_data(metadata, data, output_dir, args.format)

    logging.info("Conversion complete.")


if __name__ == "__main__":
    main()
