# -*- coding: utf-8 -*-

from array import array as arr
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import struct
from typing import Protocol

import numpy as np

from pingdata.fread import fread_data

GENERIC_WATER_TYPES = {
    1: {"water_type": "fresh", "salinity": 1.0},
    2: {"water_type": "shallow salt", "salinity": 30.0},
    3: {"water_type": "deep salt", "salinity": 35.0},
}
HELIX_WATER_TYPES = {
    0: {"water_type": "fresh", "salinity": 1.0},
    1: {"water_type": "deep salt", "salinity": 35.0},
    2: {"water_type": "shallow salt", "salinity": 30.0},
}


class MetadataType(Enum):
    GENERIC = 100
    HELIX = 64
    SOLIX = 96

    @classmethod
    def from_file_length(cls, length):
        """Returns the corresponding FileType based on file length."""
        return cls(length) if length in cls._value2member_map_ else None


class MetadataPointers(Protocol): ...


@dataclass(frozen=True)
class GenericMetadataPointers:
    endianness: str = "<i"
    SP1: list[int] = field(default_factory=lambda: [0, 0, 1, -1])
    water_code: list[int] = field(default_factory=lambda: [1, 0, 1, -1])
    SP2: list[int] = field(default_factory=lambda: [2, 0, 1, -1])
    unknown_1: list[int] = field(default_factory=lambda: [3, 0, 1, -1])
    sonar_name: list[int] = field(default_factory=lambda: [4, 0, 4, -1])
    unknown_2: list[int] = field(default_factory=lambda: [8, 0, 4, -1])
    unknown_3: list[int] = field(default_factory=lambda: [12, 0, 4, -1])
    unknown_4: list[int] = field(default_factory=lambda: [16, 0, 4, -1])
    unix_time: list[int] = field(default_factory=lambda: [20, 0, 4, -1])
    utm_e: list[int] = field(default_factory=lambda: [24, 0, 4, -1])
    utm_n: list[int] = field(default_factory=lambda: [28, 0, 4, -1])
    filename: list[int] = field(default_factory=lambda: [32, 0, 12, -1])
    numrecords: list[int] = field(default_factory=lambda: [44, 0, 4, -1])
    recordlens_ms: list[int] = field(default_factory=lambda: [48, 0, 4, -1])
    linesize: list[int] = field(default_factory=lambda: [52, 0, 4, -1])
    unknown_5: list[int] = field(default_factory=lambda: [56, 0, 4, -1])
    unknown_6: list[int] = field(default_factory=lambda: [60, 0, 4, -1])
    unknown_7: list[int] = field(default_factory=lambda: [64, 0, 4, -1])
    unknown_8: list[int] = field(default_factory=lambda: [68, 0, 4, -1])
    unknown_9: list[int] = field(default_factory=lambda: [72, 0, 4, -1])
    unknown_10: list[int] = field(default_factory=lambda: [76, 0, 4, -1])
    unknown_11: list[int] = field(default_factory=lambda: [80, 0, 4, -1])
    unknown_12: list[int] = field(default_factory=lambda: [84, 0, 4, -1])
    unknown_13: list[int] = field(default_factory=lambda: [88, 0, 4, -1])
    unknown_14: list[int] = field(default_factory=lambda: [92, 0, 4, -1])
    unknown_15: list[int] = field(default_factory=lambda: [96, 0, 4, -1])


@dataclass(frozen=True)
class HelixMetadataPointers:
    endianness: str = ">i"
    SP1: list[int] = field(default_factory=lambda: [0, 0, 1, -1])
    water_code: list[int] = field(default_factory=lambda: [1, 0, 1, -1])
    SP2: list[int] = field(default_factory=lambda: [2, 0, 1, -1])
    unknown_1: list[int] = field(default_factory=lambda: [3, 0, 1, -1])
    sonar_name: list[int] = field(default_factory=lambda: [4, 0, 4, -1])
    unknown_2: list[int] = field(default_factory=lambda: [8, 0, 4, -1])
    unknown_3: list[int] = field(default_factory=lambda: [12, 0, 4, -1])
    unknown_4: list[int] = field(default_factory=lambda: [16, 0, 4, -1])
    unix_time: list[int] = field(default_factory=lambda: [20, 0, 4, -1])
    utm_e: list[int] = field(default_factory=lambda: [24, 0, 4, -1])
    utm_n: list[int] = field(default_factory=lambda: [28, 0, 4, -1])
    filename: list[int] = field(default_factory=lambda: [32, 0, 10, -1])
    unknown_5: list[int] = field(default_factory=lambda: [42, 0, 2, -1])
    numrecords: list[int] = field(default_factory=lambda: [44, 0, 4, -1])
    recordlens_ms: list[int] = field(default_factory=lambda: [48, 0, 4, -1])
    linesize: list[int] = field(default_factory=lambda: [52, 0, 4, -1])
    unknown_6: list[int] = field(default_factory=lambda: [56, 0, 4, -1])
    unknown_7: list[int] = field(default_factory=lambda: [60, 0, 4, -1])


@dataclass(frozen=True)
class SolixMetadataPointers:
    endianness: str = "<i"
    SP1: list[int] = field(default_factory=lambda: [0, 0, 1, -1])
    water_code: list[int] = field(default_factory=lambda: [1, 0, 1, -1])
    SP2: list[int] = field(default_factory=lambda: [2, 0, 1, -1])
    unknown_1: list[int] = field(default_factory=lambda: [3, 0, 1, -1])
    sonar_name: list[int] = field(default_factory=lambda: [4, 0, 4, -1])
    unknown_2: list[int] = field(default_factory=lambda: [8, 0, 4, -1])
    unknown_3: list[int] = field(default_factory=lambda: [12, 0, 4, -1])
    unknown_4: list[int] = field(default_factory=lambda: [16, 0, 4, -1])
    unix_time: list[int] = field(default_factory=lambda: [20, 0, 4, -1])
    utm_e: list[int] = field(default_factory=lambda: [24, 0, 4, -1])
    utm_n: list[int] = field(default_factory=lambda: [28, 0, 4, -1])
    filename: list[int] = field(default_factory=lambda: [32, 0, 12, -1])
    numrecords: list[int] = field(default_factory=lambda: [44, 0, 4, -1])
    recordlens_ms: list[int] = field(default_factory=lambda: [48, 0, 4, -1])
    linesize: list[int] = field(default_factory=lambda: [52, 0, 4, -1])
    unknown_5: list[int] = field(default_factory=lambda: [56, 0, 4, -1])
    unknown_6: list[int] = field(default_factory=lambda: [60, 0, 4, -1])
    unknown_7: list[int] = field(default_factory=lambda: [64, 0, 4, -1])
    unknown_8: list[int] = field(default_factory=lambda: [68, 0, 4, -1])
    unknown_9: list[int] = field(default_factory=lambda: [72, 0, 4, -1])
    unknown_10: list[int] = field(default_factory=lambda: [76, 0, 4, -1])
    unknown_11: list[int] = field(default_factory=lambda: [80, 0, 4, -1])
    unknown_12: list[int] = field(default_factory=lambda: [84, 0, 4, -1])
    unknown_13: list[int] = field(default_factory=lambda: [88, 0, 4, -1])
    unknown_14: list[int] = field(default_factory=lambda: [92, 0, 4, -1])


@dataclass
class Metadata:
    endianness: str
    water_code: int
    sonar_name: str
    unix_time: int
    utm_e: int
    utm_n: int
    filename: str
    numrecords: int
    recordlens_ms: int
    linesize: int
    SP1: int
    SP2: int
    unknown_1: int
    unknown_2: int
    unknown_3: int
    unknown_4: int
    unknown_5: int
    unknown_6: int
    unknown_7: int
    unknown_8: int
    unknown_9: int
    unknown_10: int
    unknown_11: int
    unknown_12: int
    unknown_13: int
    unknown_14: int
    water_type: str | None = None
    temperature: float | None = None
    salinity: float | None = None
    sound_speed: float | None = None
    time_varying_gain: float | None = None
    pixel_size: float | None = None
    file_path: Path | None = None


def read_metadata(
    file: Path, temperature: float = 10.0, sound_speed: float | None = None
) -> Metadata:
    file_size_bytes = file.stat().st_size
    file_type = MetadataType.from_file_length(file_size_bytes)
    metadata = reader_factory(file_type)(file)

    # Assign salinity based on water_code:
    water_code = metadata["water_code"]
    metadata.update(assign_water_type(file_type, water_code))

    # Compute sound speed and time-varying gain:
    if sound_speed is None:
        sound_speed = compute_sound_speed(temperature, metadata["salinity"])
    time_varying_gain = compute_time_varying_gain(sound_speed)
    metadata.update(
        {
            "temperature": temperature,
            "sound_speed": sound_speed,
            "time_varying_gain": time_varying_gain,
        }
    )
    # Compute pixel size:
    # TODO: Figure out how to add frequency
    pixel_size = compute_pixel_size(
        sound_speed, transducer_length=0.108, frequency=200e3
    )
    metadata.update({"pixel_size": pixel_size})

    # Add file metadata filepath:
    metadata.update({"file_path": file})

    return Metadata(**metadata)


def read_metadata_generic(file: Path) -> dict:
    # Implementation for reading generic metadata
    return read_with_pointers(file, GenericMetadataPointers())


def read_metadata_helix(file: Path) -> dict:
    # Implementation for reading Helix metadata
    return read_with_pointers(file, HelixMetadataPointers())


def read_metadata_solix(file: Path) -> dict:
    # Implementation for reading Solix metadata
    return read_with_pointers(file, SolixMetadataPointers())


def read_with_pointers(file: Path, pointers: MetadataPointers) -> dict:
    data = {}
    endian = pointers.endianness
    with file.open("rb") as f:
        for key, value in asdict(pointers).items():
            if key == "endianness":
                data[key] = value
                continue
            f.seek(value[0])
            if value[2] == 4:
                byte = struct.unpack(
                    endian, arr("B", fread_data(f, value[2], "B")).tobytes()
                )[
                    0
                ]  # Decode byte
            if value[2] < 4:
                byte = fread_data(f, value[2], "B")[0]  # Decode byte
            if value[2] > 4:
                byte = (
                    arr("B", fread_data(f, value[2], "B")).tobytes().decode()
                )  # Decode byte

            data[key] = byte

    return data


def reader_factory(file_type: str) -> callable:
    if file_type == MetadataType.GENERIC:
        return read_metadata_generic
    if file_type == MetadataType.HELIX:
        return read_metadata_helix
    if file_type == MetadataType.SOLIX:
        return read_metadata_solix
    raise ValueError(f"Unsupported file type for metadata reading.")


def assign_water_type(file_type: str, water_code: int) -> dict:
    if file_type == MetadataType.GENERIC:
        return water_type_generic(water_code)
    if file_type == MetadataType.HELIX:
        return water_type_helix(water_code)
    if file_type == MetadataType.SOLIX:
        return water_type_solix(water_code)


def water_type_generic(water_code: int) -> dict:
    return GENERIC_WATER_TYPES.get(
        water_code, {"water_type": "unknown", "salinity": 0.0}
    )


def water_type_helix(water_code: int) -> dict:
    return HELIX_WATER_TYPES.get(water_code, {"water_type": "unknown", "salinity": 0.0})


def water_type_solix(water_code: int) -> dict:
    return water_type_generic(water_code)


def compute_pixel_size(
    sound_speed: float, transducer_length: float = 0.108, frequency: float = 455e3
) -> float:
    # Theta at 3 dB in the horizontal:
    theta3db = np.arcsin(sound_speed / (transducer_length * frequency))
    # Pixel size (m):
    return float(2 / (np.pi * theta3db))


def compute_sound_speed(temp: float, salinity: float) -> float:
    return (
        1449.05
        + 45.7 * temp
        - 5.21 * temp**2
        + 0.23 * temp**3
        + (1.333 - 0.126 * temp + 0.009 * temp**2) * (salinity - 35)
    )


def compute_time_varying_gain(sound_speed: float) -> float:
    return ((8.5e-5) + (3 / 76923) + ((8.5e-5) / 4)) * sound_speed
