# -*- coding: utf-8 -*-

import numpy as np

from pingdata.beam_data import Beam, PingMetadata


def read_sonar_data(metadata: PingMetadata) -> list[np.ndarray]:
    data = {}
    for beam in metadata.beams:
        beam_data = read_sonar_beam(beam)
        data[f"{beam.type.name}_{beam.type}"] = {
            "time_start": beam.metadata["timestamp_utc"][0],
            "data": beam_data,
            "time_s": beam.metadata["time_s"].to_numpy(),
        }

    return data


def read_sonar_beam(beam: Beam) -> np.ndarray:
    columns_to_keep = ["ping_cnt", "index", "son_offset"]
    metadata = beam.metadata[columns_to_keep]

    pointers = metadata["index"].to_list()
    header_lengths = metadata["son_offset"].to_list()
    buffer_lengths = metadata["ping_cnt"].to_list()

    data = np.zeros((len(pointers), max(buffer_lengths)), dtype=">u1")
    with beam.file.open("rb") as f:
        for i, (pointer, header_length, buffer_length) in enumerate(
            zip(pointers, header_lengths, buffer_lengths)
        ):
            # Read the sonar data
            f.seek(pointer + header_length)
            buffer = f.read(buffer_length)
            data[i, : len(buffer)] = np.frombuffer(buffer, dtype=">u1")

    return data
