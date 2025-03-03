# -*- coding: utf-8 -*-

from array import array as arr
from dataclasses import asdict, dataclass
from enum import Enum, StrEnum
from pathlib import Path
import struct
from typing import BinaryIO, Protocol, Sequence

import numpy as np
import pandas as pd
import pyproj

from pingdata import structures


class BeamType(StrEnum):
    B000 = "ds_lowfreq"
    B001 = "ds_highfreq"
    B002 = "ss_port"
    B003 = "ss_star"
    B004 = "ds_vhighfreq"
    UNKNOWN = "unknown"

    @classmethod
    def from_beam(cls, beam: str) -> "BeamType":
        return cls.__members__.get(beam, cls.UNKNOWN)


@dataclass
class Beam:
    channel: int
    type: str
    file: Path
    metadata: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self.output_file = (
            self.file.parent / f"B{self.channel:03d}_{self.type}_metadata.csv"
        )


class MetadataType(Enum):
    GENERIC = 100
    HELIX = 64
    SOLIX = 96

    @classmethod
    def from_file_length(cls, length):
        """Returns the corresponding FileType based on file length."""
        return cls(length) if length in cls._value2member_map_ else None


class MetadataPointers(Protocol): ...


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


@dataclass
class PingMetadata:
    metadata: Metadata | None = None
    beams: list[pd.DataFrame] | None = None


def assign_water_type(file_type: str, water_code: int) -> dict:
    if file_type == MetadataType.GENERIC:
        return water_type_generic(water_code)
    if file_type == MetadataType.HELIX:
        return water_type_helix(water_code)
    if file_type == MetadataType.SOLIX:
        return water_type_solix(water_code)


def beam_files(path: Path):
    stem = path.stem
    return (path.parent / stem).rglob("*.SON")


def compute_along_track_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate along track distance based on time ellapsed and gps speed.
    """

    ts = df["time_s"].to_numpy()
    ss = df["speed_ms"].to_numpy()
    ds = np.zeros((len(ts)))

    # Offset arrays for faster calculation
    ts1 = ts[1:]
    ss1 = ss[1:]
    ts = ts[:-1]

    # Calculate instantaneous distance
    d = (ts1 - ts) * ss1
    ds[1:] = d

    # Accumulate distance
    ds = np.cumsum(ds)

    df["trk_dist"] = ds
    return df


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


def condition_ping_metadata_dataframe(
    df: pd.DataFrame, pixel_size: float, unix_time: int
) -> pd.DataFrame:
    """ """

    # Calculate range
    df["max_range"] = df["ping_cnt"] * pixel_size

    # Convert frequency (Hz) to kHz
    df["f_kHz"] = df["f_kHz"] / 1000

    # Easting and northing variances appear to be stored in the file
    ## They are reported in cm's so need to convert
    df["e_err_m"] = np.abs(df["e_err_m"]) / 100
    df["n_err_m"] = np.abs(df["n_err_m"]) / 100

    # Now calculate hdop from n/e variances
    df["hdop"] = np.round(np.sqrt(df["e_err_m"] + df["n_err_m"]), 2)

    # Convert eastings/northings to latitude/longitude (from Py3Hum - convert using International 1924 spheroid)
    lat, lon = convert_epsg3395_to_latlon(
        eastings=df["utm_e"].to_numpy(),
        northings=df["utm_n"].to_numpy(),
    )
    df["lon"] = lon
    df["lat"] = lat

    # Instrument heading, speed, and depth need to be divided by 10 to be
    # in appropriate units.
    df["instr_heading"] = df["instr_heading"] / 10
    df["speed_ms"] = df["speed_ms"] / 10
    df["inst_dep_m"] = df["inst_dep_m"] / 10

    # Get units into appropriate format
    df["time_s"] = df["time_s"] / 1000  # milliseconds to seconds
    # df["tempC"] = temperature
    # # Can we figure out a way to base transducer length on where we think the recording came from?
    # df['t'] = 0.108
    # Use recording unix time to calculate each sonar records unix time
    try:
        # starttime = float(df['unix_time'])
        starttime = float(unix_time)
        df["caltime"] = starttime + df["time_s"]

    except:
        df["caltime"] = 0

    df["timestamp_utc"] = pd.to_datetime(df["caltime"], unit="s").astype(str)

    df = df.drop("caltime", axis=1)

    # df = df[df["e"] != np.inf]
    df = df[df["record_num"] >= 0]

    lastIdx = df["index"].iloc[-1]
    df = df[df["index"] <= lastIdx]

    # Calculate along-track distance from 'time's and 'speed_ms'. Approximate distance estimate
    df = compute_along_track_distance(df)

    return df


def convert_epsg3395_to_latlon(
    eastings: Sequence, northings: Sequence
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts EPSG 3395 (World Mercator) Easting/Northing to Latitude/Longitude (WGS84).

    Parameters:
    - eastings (array-like): Easting values in EPSG 3395.
    - northings (array-like): Northing values in EPSG 3395.

    Returns:
    - (numpy.ndarray, numpy.ndarray): (latitudes, longitudes) in decimal degrees (WGS84).
    """
    # Define a transformation from EPSG 3395 (World Mercator) to EPSG 4326 (WGS84 Lat/Lon)
    transformer = pyproj.Transformer.from_crs("EPSG:3395", "EPSG:4326", always_xy=True)

    # Perform the transformation
    longitudes, latitudes = transformer.transform(eastings, northings)

    return np.array(latitudes), np.array(longitudes)


def convert_to_df(
    data: list[dict],
    frame_offset,
    header_bytes,
    chunk_id,
    nchunk: int,
    chunk: int,
    metadata: Metadata,
) -> pd.DataFrame:

    df = pd.DataFrame.from_dict(data)

    # Add in the frame offset
    df["index"] = frame_offset

    # Add in the son_offset (headBytes for Humminbird)
    df["son_offset"] = header_bytes

    # Add chunk id
    df["chunk_id"] = chunk_id

    # Do unit conversions
    df = condition_ping_metadata_dataframe(df, metadata.pixel_size, metadata.unix_time)

    # Drop spacer and unknown columns
    for col in df.columns:
        if "SP" in col:
            df.drop(col, axis=1, inplace=True)
        if "unknown" in col:
            df.drop(col, axis=1, inplace=True)

    # Drop head_start
    df.drop("head_start", axis=1, inplace=True)
    df.drop("head_end", axis=1, inplace=True)

    # Update last chunk if too small (for rectification)
    lastChunk = df[df["chunk_id"] == chunk]
    if len(lastChunk) <= nchunk / 2:
        df.loc[df["chunk_id"] == chunk, "chunk_id"] = chunk - 1

    return df


def fread_data(binary_data: BinaryIO, num_bytes: int, byte_type: str) -> list:
    data = arr(byte_type)
    data.fromfile(binary_data, num_bytes)
    return list(data)


def get_beams(path: Path) -> list[Beam]:
    files = beam_files(path)

    beams = []
    for i, file in enumerate(sorted(list(files))):
        beams.append(
            Beam(
                channel=i + 1,
                type=BeamType.from_beam(file.stem),
                file=file,
            )
        )

    return beams


def get_head_structure(header_length: int) -> np.dtypes.VoidDType:
    """Returns an instance of the appropriate header dataclass."""
    header_structure = structures.PING_HEADER_REGISTRY.get(header_length)
    if not header_structure:
        raise ValueError(f"Unsupported header size: {header_length}")
    return header_structure


def get_ping_header_length(f: BinaryIO) -> int:
    """
    Determine .SON ping header length based on known Humminbird
    .SON file structure.  Humminbird stores sonar records in packets, where
    the first x bytes of the packet contain metadata (record number, northing,
    easting, time elapsed, depth, etc.), proceeded by the sonar/ping returns
    associated with that ping.  This function will search the first
    ping to determine the length of ping header.

    ----------------------------
    Required Pre-processing step
    ----------------------------
    __init__()

    -------
    Returns
    -------
    headBytes, indicating length, in bytes, of ping header.

    --------------------
    Next Processing Step
    --------------------
    _getHeadStruct()
    """

    HEADER_END_LOCATION: int = 33

    header_bytes = 0  # Counter to track sonar header length
    foundEnd = False  # Flag to track if end of sonar header found
    while foundEnd is False and header_bytes < 200:
        lastPos = f.tell()  # Get current position in file (byte offset from beginning)
        byte = fread_data(f, 1, "B")  # Decode byte value

        # Check if we found the end of the ping.
        ## A value of 33 **may** indicate end of ping.
        if byte[0] == HEADER_END_LOCATION and lastPos > 3:
            # Double check we found the actual end by moving backward -6 bytes
            ## to see if value is 160 (spacer preceding number of ping records)
            f.seek(-6, 1)
            byte = fread_data(f, 1, "B")
            if byte[0] == 160:
                foundEnd = True
            else:
                # Didn't find the end of header
                # Move cursor back to lastPos+1
                f.seek(lastPos + 1)
        else:
            # Haven't found the end
            pass
        header_bytes += 1

    # i reaches 200, then we have exceeded known Humminbird header length.
    ## Set i to 0, then the next sonFile will be checked.
    if header_bytes == 200:
        header_bytes = 0

    # headBytes = i # Store data in class attribute for later use
    return header_bytes


def read_beams(metadata: Metadata) -> list[Beam]:
    beams = get_beams(metadata.file_path)

    for beam in beams:
        beam.metadata = read_metadata_all_pings(beam.file, metadata)

    return beams


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


def read_metadata_all_pings(
    file: Path,
    metadata: Metadata,
    nchunk: int = 500,
) -> pd.DataFrame:
    """ """

    # Get file length
    file_length = file.stat().st_size

    # Initialize counter
    pointer = 0
    chunk_i = 0
    chunk = 0

    header_data_all = []
    frame_offset = []
    chunk_id = []

    with file.open("rb") as f:

        header_length = get_ping_header_length(f)
        header_structure = get_head_structure(header_length)

        # Decode ping header
        while pointer < file_length:
            # Get header data at offset i
            header_data, new_pointer = read_metadata_single_ping(
                f, pointer, header_length, header_structure
            )

            # Add frame offset
            frame_offset.append(pointer)
            header_data_all.append(header_data)
            chunk_id.append(chunk)

            # update counter with current position
            pointer = new_pointer

            if chunk_i == nchunk:
                chunk_i = 0
                chunk += 1
            else:
                chunk_i += 1

    return convert_to_df(
        header_data_all, frame_offset, header_length, chunk_id, nchunk, chunk, metadata
    )


def read_metadata_single_ping(
    f: BinaryIO, pointer: int, header_length: int, header_structure: np.dtypes.VoidDType
) -> tuple[dict, int]:
    # Get necessary attributes

    # Move to offset
    f.seek(pointer)

    # Get the data
    buffer = f.read(header_length)

    # Read the data
    raw_header = np.frombuffer(buffer, dtype=header_structure)

    header = {}
    for name in raw_header.dtype.fields.keys():
        header[name] = raw_header[name][0].item()

    # Next ping header is from current position + ping_cnt
    new_pointer = f.tell() + raw_header[0][-2]

    return header, new_pointer


def read_metadata_generic(file: Path) -> dict:
    # Implementation for reading generic metadata
    return read_metadata_with_pointers(file, structures.GenericMetadataPointers())


def read_metadata_helix(file: Path) -> dict:
    # Implementation for reading Helix metadata
    return read_metadata_with_pointers(file, structures.HelixMetadataPointers())


def read_metadata_solix(file: Path) -> dict:
    # Implementation for reading Solix metadata
    return read_metadata_with_pointers(file, structures.SolixMetadataPointers())


def read_metadata_with_pointers(file: Path, pointers: MetadataPointers) -> dict:
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


def read_ping_metadata(
    file: Path, temperature: float, sound_speed: float | None = None
) -> PingMetadata:
    metadata = read_metadata(file, temperature=temperature)
    beams = read_beams(metadata)
    return PingMetadata(metadata=metadata, beams=beams)


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


def reader_factory(file_type: str) -> callable:
    if file_type == MetadataType.GENERIC:
        return read_metadata_generic
    if file_type == MetadataType.HELIX:
        return read_metadata_helix
    if file_type == MetadataType.SOLIX:
        return read_metadata_solix
    raise ValueError(f"Unsupported file type for metadata reading.")


def water_type_generic(water_code: int) -> dict:
    return structures.GENERIC_WATER_TYPES.get(
        water_code, {"water_type": "unknown", "salinity": 0.0}
    )


def water_type_helix(water_code: int) -> dict:
    return structures.HELIX_WATER_TYPES.get(
        water_code, {"water_type": "unknown", "salinity": 0.0}
    )


def water_type_solix(water_code: int) -> dict:
    return water_type_generic(water_code)
