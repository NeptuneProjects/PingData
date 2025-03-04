# -*- coding: utf-8 -*-

from array import array as arr
from dataclasses import asdict, dataclass
from enum import Enum, StrEnum
from pathlib import Path
import struct
from typing import BinaryIO, Generator, Protocol, Sequence

import numpy as np
import pandas as pd
import pyproj
from scipy.io import savemat

from pingdata import structures


class BeamType(StrEnum):
    """Enumeration of beam types."""

    B000 = "ds_lowfreq"
    B001 = "ds_highfreq"
    B002 = "ss_port"
    B003 = "ss_star"
    B004 = "ds_vhighfreq"
    UNKNOWN = "unknown"

    @classmethod
    def from_beam(cls, beam: str) -> "BeamType":
        return cls.__members__.get(beam, cls.UNKNOWN)


class MetadataType(Enum):
    """Enumeration of metadata types."""

    GENERIC = 100
    HELIX = 64
    SOLIX = 96

    @classmethod
    def from_file_length(cls, length) -> "MetadataType":
        """Returns the corresponding FileType based on file length."""
        return cls(length) if length in cls._value2member_map_ else None


class MetadataPointers(Protocol): ...


@dataclass
class Beam:
    """Container for beam file (SON)."""

    channel: int
    type: str
    file: Path
    metadata: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self.output_file = (
            self.file.parent / f"B{self.channel:03d}_{self.type}_metadata.csv"
        )


@dataclass
class Metadata:
    """Container for metadata file (DAT)."""

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
    """Container for ping metadata."""

    metadata: Metadata | None = None
    beams: list[pd.DataFrame] | None = None


def assign_water_type(file_type: str, water_code: int) -> dict:
    """Assign water type and salinity based on the file type and water code.

    Args:
        file_type: Type of the metadata file.
        water_code: Water code from the metadata.

    Returns:
        Dictionary containing the water type and salinity.
    """
    if file_type == MetadataType.GENERIC:
        return water_type_generic(water_code)
    if file_type == MetadataType.HELIX:
        return water_type_helix(water_code)
    if file_type == MetadataType.SOLIX:
        return water_type_solix(water_code)


def beam_files(path: Path) -> Generator[Path, None, None]:
    """Get beam files from the given path.

    Args:
        path: Path to the directory containing beam files.

    Returns:
        Generator object yielding beam files.
    """
    stem = path.stem
    return (path.parent / stem).rglob("*.SON")


def compute_along_track_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate along track distance based on time ellapsed and GPS speed.

    Args:
        df: DataFrame containing time and speed columns.

    Returns:
        DataFrame with an additional column for track distance.
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
    """Compute pixel size using sound speed and geometry.

    Args:
        sound_speed: Speed of sound in water (m/s).
        transducer_length: Length of the transducer (m).
        frequency: Frequency of the sonar (Hz).

    Returns:
        Pixel size (m).
    """
    # Theta at 3 dB in the horizontal:
    theta3db = np.arcsin(sound_speed / (transducer_length * frequency))
    # Pixel size (m):
    return float(2 / (np.pi * theta3db))


def compute_sound_speed(temp: float, salinity: float) -> float:
    """Compute sound speed in seawater using temperature and salinity.

    Args:
        temp: Temperature (C).
        salinity: Salinity (ppt).

    Returns:
        Sound speed (m/s).
    """
    return (
        1449.05
        + 45.7 * temp
        - 5.21 * temp**2
        + 0.23 * temp**3
        + (1.333 - 0.126 * temp + 0.009 * temp**2) * (salinity - 35)
    )


def compute_time_varying_gain(sound_speed: float) -> float:
    """Compute time-varying gain based on sound speed.

    Args:
        sound_speed: Speed of sound in water (m/s).

    Returns:
        Time-varying gain (dB).
    """
    return ((8.5e-5) + (3 / 76923) + ((8.5e-5) / 4)) * sound_speed


def condition_ping_metadata_dataframe(
    df: pd.DataFrame, pixel_size: float, unix_time: int
) -> pd.DataFrame:
    """Condition the ping metadata DataFrame.

    Args:
        df: DataFrame containing ping metadata.
        pixel_size: Pixel size (m).
        unix_time: Unix time of the recording.

    Returns:
        DataFrame with conditioned metadata.
    """
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


def construct_output_dir(input_file: Path, output_dir: Path | None = None) -> Path:
    """Get the save path for the converted data.

    If `output_dir` is not provided, the output directory will be the same as
    the input file's directory, and the filename will be the same as the input
    file's name without the extension.

    Args:
        input_file: Path to the input file.
        output_dir: Path to the output directory. Defaults to None.

    Returns:
        Path to the output directory.
    """
    if not output_dir:
        return input_file.parent / input_file.stem
    return output_dir / input_file.stem


def convert_data(
    input_file: Path, temperature: float = 10.0, sound_speed: float = 1500.0
) -> tuple[PingMetadata, dict[dict]]:
    """Convert sonar data to a specified format.

    Args:
        input_file: Path to the input file.
        temperature: Water temperature in Celsius. Defaults to 10.0.
        sound_speed: Sound speed in m/s. Defaults to 1500.0.

    Returns:
        Tuple containing the ping metadata and the sonar data.
    """
    ping_metadata = read_ping_metadata(input_file, temperature, sound_speed)
    return ping_metadata, read_sonar_data(ping_metadata)


def convert_epsg3395_to_latlon(
    eastings: Sequence, northings: Sequence
) -> tuple[np.ndarray, np.ndarray]:
    """Convert EPSG 3395 (World Mercator) coordinates to latitude and longitude.

    Args:
        eastings: Easting coordinates.
        northings: Northing coordinates.

    Returns:
        Tuple of latitude and longitude arrays.
    """
    # Define a transformation from EPSG 3395 (World Mercator) to EPSG 4326 (WGS84 Lat/Lon)
    transformer = pyproj.Transformer.from_crs("EPSG:3395", "EPSG:4326", always_xy=True)

    # Perform the transformation
    longitudes, latitudes = transformer.transform(eastings, northings)

    return np.array(latitudes), np.array(longitudes)


def convert_to_df(
    data: list[dict],
    frame_offset: list[int],
    header_bytes: list[int],
    chunk_id: list[int],
    nchunk: int,
    chunk: int,
    metadata: Metadata,
) -> pd.DataFrame:
    """Convert the list of dictionaries to a DataFrame and condition it.

    Args:
        data: List of dictionaries containing metadata.
        frame_offset: Frame offset for each record.
        header_bytes: Header bytes for each record.
        chunk_id: Chunk ID for each record.
        nchunk: Number of chunks.
        chunk: Current chunk number.
        metadata: Metadata object.

    Returns:
        DataFrame containing the conditioned metadata.
    """

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
    """Read data from a binary file.

    Args:
        binary_data: Binary file object.
        num_bytes: Number of bytes to read.
        byte_type: Type of data to read (e.g., 'B', 'H', 'I', etc.).

    Returns:
        List of data read from the file.
    """
    data = arr(byte_type)
    data.fromfile(binary_data, num_bytes)
    return list(data)


def get_beams(path: Path) -> list[Beam]:
    """Get a list of beams from the given path.

    Args:
        path: Path to the directory containing beam files.

    Returns:
        List of Beam objects.
    """
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
    """Get the structure of the ping header based on its length.

    Args:
        header_length: Length of the ping header.

    Returns:
        The structure of the ping header as a numpy dtype.
    """
    header_structure = structures.PING_HEADER_REGISTRY.get(header_length)
    if not header_structure:
        raise ValueError(f"Unsupported header size: {header_length}")
    return header_structure


def get_ping_header_length(f: BinaryIO) -> int:
    """Determine .SON ping header length based on known Humminbird
    .SON file structure.

    Humminbird stores sonar records in packets, where the first x bytes
    of the packet contain metadata (record number, northing, easting, time
    elapsed, depth, etc.), proceeded by the sonar/ping returns associated
    with that ping. This function will search the first ping to determine
    the length of ping header.

    Args:
        f: Binary file object.

    Returns:
        int: Length of the ping header in bytes.
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
    """Read beams from the given metadata.

    Args:
        metadata: Metadata object containing file path.

    Returns:
        List of Beam objects.
    """
    beams = get_beams(metadata.file_path)

    for beam in beams:
        beam.metadata = read_metadata_all_pings(beam.file, metadata)

    return beams


def read_metadata(
    file: Path, temperature: float = 10.0, sound_speed: float | None = None
) -> Metadata:
    """Read metadata from the given file.

    Args:
        file: Path to the metadata file.
        temperature: Temperature in Celsius.
        sound_speed: Sound speed in m/s.

    Returns:
        Metadata object containing the metadata.
    """
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
    """Read metadata from all pings in the given file.

    Args:
        file: Path to the metadata file.
        metadata: Metadata object containing file path.
        nchunk: Number of chunks to read.

    Returns:
        DataFrame containing the metadata for all pings.
    """
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
    """Read metadata from a single ping.

    Args:
        f: Binary file object.
        pointer: Current position in the file.
        header_length: Length of the ping header.
        header_structure: Structure of the ping header.

    Returns:
        Tuple containing the header data and the new pointer position.
    """
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
    """Read metadata from a generic file.

    Args:
        file: Path to the metadata file.

    Returns:
        Dictionary containing the metadata.
    """
    return read_metadata_with_pointers(file, structures.GenericMetadataPointers())


def read_metadata_helix(file: Path) -> dict:
    """Read metadata from a Helix file.

    Args:
        file: Path to the metadata file.

    Returns:
        Dictionary containing the metadata.
    """
    return read_metadata_with_pointers(file, structures.HelixMetadataPointers())


def read_metadata_solix(file: Path) -> dict:
    """Read metadata from a Solix file.

    Args:
        file: Path to the metadata file.

    Returns:
        Dictionary containing the metadata.
    """
    return read_metadata_with_pointers(file, structures.SolixMetadataPointers())


def read_metadata_with_pointers(file: Path, pointers: MetadataPointers) -> dict:
    """Read metadata from a file using the given pointers.

    Args:
        file: Path to the metadata file.
        pointers: MetadataPointers object containing the pointers.

    Returns:
        Dictionary containing the metadata.
    """
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
    """Read ping metadata from the given file.

    Args:
        file: Path to the metadata file.
        temperature: Temperature in Celsius.
        sound_speed: Sound speed in m/s.

    Returns:
        PingMetadata object containing the metadata and beams.
    """
    metadata = read_metadata(file, temperature=temperature)
    beams = read_beams(metadata)
    return PingMetadata(metadata=metadata, beams=beams)


def read_sonar_beam(beam: Beam) -> np.ndarray:
    """Read sonar data from the given beam.

    Args:
        beam: Beam object containing the file path.

    Returns:
        Numpy array containing the sonar data.
    """
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


def read_sonar_data(metadata: PingMetadata) -> dict[dict]:
    """Read sonar data from the given metadata.

    Args:
        metadata: PingMetadata object containing the metadata and beams.

    Returns:
        Dictionary containing the sonar data for each beam.
    """
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
    """Factory function to get the appropriate metadata reader based on file type.

    Args:
        file_type: Type of the metadata file.

    Returns:
        Function to read metadata for the specified file type.
    """
    if file_type == MetadataType.GENERIC:
        return read_metadata_generic
    if file_type == MetadataType.HELIX:
        return read_metadata_helix
    if file_type == MetadataType.SOLIX:
        return read_metadata_solix
    raise ValueError(f"Unsupported file type for metadata reading.")


def save_data(
    metadata: PingMetadata, data: dict, output_dir: Path, file_format: str
) -> None:
    """Save data from each beam to the specified format.

    Args:
        metadata: Metadata of the ping data.
        data: Sonar data to be saved.
        output_dir: Directory to save the files.
        file_format: The desired file format.
    """
    save_func = save_function_factory(file_format)
    for beam, (beam_name, beam_data) in zip(metadata.beams, data.items()):
        beam.metadata.to_csv(output_dir / f"{beam_name}.csv", index=False)
        save_func(beam_name, beam_data, output_dir)


def save_function_factory(file_format: str):
    """Factory function to save data in different formats.

    Args:
        file_format: The desired file format.

    Returns:
        A function that saves data in the specified format.

    Raises:
        ValueError: If the specified file format is not supported.
    """
    if file_format == "hdf5":
        return save_hdf5
    if file_format == "mat":
        return save_mat
    if file_format == "npy":
        return save_npy
    raise ValueError(f"Unsupported file format: {file_format}")


def save_hdf5(beam_name: str, beam_data: dict, output_dir: Path) -> None:
    """Save beam data to a .hdf5 file.

    Args:
        beam_name: Name of the beam.
        beam_data: Beam data to be saved.
        output_dir: Directory to save the file.
    """
    import h5py

    with h5py.File(output_dir / f"{beam_name}.hdf5", "w") as f:
        for key, value in beam_data.items():
            f.create_dataset(key, data=value)


def save_mat(beam_name: str, beam_data: dict, output_dir: Path) -> None:
    """Save beam data to a .mat file.

    Args:
        beam_name: Name of the beam.
        beam_data: Beam data to be saved.
        output_dir: Directory to save the file.
    """
    savemat(
        output_dir / f"{beam_name}.mat",
        {
            "data": beam_data,
        },
    )


def save_npy(beam_name: str, beam_data: dict, output_dir: Path) -> None:
    """Save beam data to a .npy file.

    Args:
        beam_name: Name of the beam.
        beam_data: Beam data to be saved.
        output_dir: Directory to save the file.
    """
    np.save(output_dir / f"{beam_name}.npy", beam_data)


def water_type_generic(water_code: int) -> dict:
    """Get the water type and salinity based on the water code.

    Args:
        water_code: Water code from the metadata.

    Returns:
        Dictionary containing the water type and salinity.
    """
    return structures.GENERIC_WATER_TYPES.get(
        water_code, {"water_type": "unknown", "salinity": 0.0}
    )


def water_type_helix(water_code: int) -> dict:
    """Get the water type and salinity based on the water code.

    Args:
        water_code: Water code from the metadata.

    Returns:
        Dictionary containing the water type and salinity.
    """
    return structures.HELIX_WATER_TYPES.get(
        water_code, {"water_type": "unknown", "salinity": 0.0}
    )


def water_type_solix(water_code: int) -> dict:
    """Get the water type and salinity based on the water code.

    Args:
        water_code: Water code from the metadata.

    Returns:
        Dictionary containing the water type and salinity.
    """
    return water_type_generic(water_code)
