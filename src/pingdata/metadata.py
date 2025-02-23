from array import array as arr
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
import struct
from typing import BinaryIO, Protocol


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


def read_with_pointers(file: Path, pointers: MetadataPointers) -> dict:
    data = {}
    endian = pointers.endianness
    with file.open("rb") as f:
        for key, value in asdict(pointers).items():
            if key == "endianness":
                continue
            f.seek(value[0])
            if value[2] == 4:
                byte = struct.unpack(
                    endian, arr("B", _fread_data(f, value[2], "B")).tobytes()
                )[
                    0
                ]  # Decode byte
            if value[2] < 4:
                byte = _fread_data(f, value[2], "B")[0]  # Decode byte
            if value[2] > 4:
                byte = (
                    arr("B", _fread_data(f, value[2], "B")).tobytes().decode()
                )  # Decode byte

            data[key] = byte

    return data


def read_metadata(file: Path) -> dict:
    file_size_bytes = file.stat().st_size
    file_type = MetadataType.from_file_length(file_size_bytes)
    metadata = reader_factory(file_type)(file)
    # Assign salinity based on water_code:
    water_code = metadata["water_code"]
    print(water_code)
    metadata.update(assign_water_type(file_type, water_code))

    return metadata


def read_generic_metadata() -> dict:
    # Implementation for reading generic metadata
    pass


def read_helix_metadata() -> dict:
    # Implementation for reading Helix metadata
    pass


def read_solix_metadata(file: Path) -> dict:
    # Implementation for reading Solix metadata
    metadata = read_with_pointers(file, SolixMetadataPointers())
    return metadata


def reader_factory(file_type: str) -> callable:
    if file_type == MetadataType.GENERIC:
        return read_generic_metadata
    if file_type == MetadataType.HELIX:
        return read_helix_metadata
    if file_type == MetadataType.SOLIX:
        return read_solix_metadata
    raise ValueError(f"Unsupported file type for metadata reading.")


def _fread_data(binary_data: BinaryIO, num_bytes: int, byte_type: str):
    data = arr(byte_type)
    data.fromfile(binary_data, num_bytes)
    return list(data)


def assign_water_type(file_type: str, water_code: int) -> dict:
    if file_type == MetadataType.GENERIC:
        return water_type_generic(water_code)
    if file_type == MetadataType.HELIX:
        return water_type_helix(water_code)
    if file_type == MetadataType.SOLIX:
        return water_type_solix(water_code)


def water_type_generic(water_code: int) -> dict:
    if water_code == 1:
        water_type = "fresh"
        salinity = 1.0
    elif water_code == 2:
        water_type = "shallow salt"
        salinity = 30.0
    elif water_code == 3:
        water_type = "deep salt"
        salinity = 35.0
    else:
        water_type = "unknown"
        salinity = 0.0
    return {"water_type": water_type, "salinity": salinity}


def water_type_helix(water_code: int) -> dict:
    if water_code == 0:
        water_type = "fresh"
        salinity = 1.0
    if water_code == 1:
        water_type = "deep salt"
        salinity = 35.0
    if water_code == 2:
        water_type = "shallow salt"
        salinity = 30.0
    else:
        water_type = "unknown"
        salinity = 0.0
    return {"water_type": water_type, "salinity": salinity}


def water_type_solix(water_code: int) -> dict:
    return water_type_generic(water_code)
