# -*- coding: utf-8 -*-

from dataclasses import dataclass
import datetime
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Protocol, Sequence

import numpy as np
import pandas as pd
import pyproj

from pingdata.metadata import Metadata, read_metadata
from pingdata.fread import fread_data


HEADER_STRUCTURE_67 = np.dtype(
    [
        ("head_start", ">u4"),
        ("SP128", ">u1"),
        ("record_num", ">u4"),
        ("SP129", ">u1"),
        ("time_s", ">u4"),
        ("SP130", ">u1"),
        ("utm_e", ">i4"),
        ("SP131", ">u1"),
        ("utm_n", ">i4"),
        ("SP132", ">u1"),
        ("gps1", ">u2"),
        ("instr_heading", ">u2"),
        ("SP133", ">u1"),
        ("gps2", ">u2"),
        ("speed_ms", ">u2"),
        ("SP135", ">u1"),
        ("inst_dep_m", ">u4"),
        ("SP80", ">u1"),
        ("beam", ">u1"),
        ("SP81", ">u1"),
        ("volt_scale", ">u1"),
        ("SP146", ">u1"),
        ("f", ">u4"),
        ("SP83", ">u1"),
        ("unknown_83", ">u1"),
        ("SP84", ">u1"),
        ("unknown_84", ">u1"),
        ("SP149", ">u1"),
        ("unknown_149", ">u4"),
        ("SP86", ">u1"),
        ("e_err_m", ">u1"),
        ("SP87", ">u1"),
        ("n_err_m", ">u1"),
        ("SP160", ">u1"),
        ("ping_cnt", ">u4"),
        ("head_end", ">u1"),
    ]
)

HEADER_STRUCTURE_72 = np.dtype(
    [
        ("head_start", ">u4"),
        ("SP128", ">u1"),
        ("record_num", ">u4"),
        ("SP129", ">u1"),
        ("time_s", ">u4"),
        ("SP130", ">u1"),
        ("utm_e", ">i4"),
        ("SP131", ">u1"),
        ("utm_n", ">i4"),
        ("SP132", ">u1"),
        ("gps1", ">u2"),
        ("instr_heading", ">u2"),
        ("SP133", ">u1"),
        ("gps2", ">u2"),
        ("speed_ms", ">u2"),
        ("SP134", ">u1"),
        ("unknown_134", ">u4"),
        ("SP135", ">u1"),
        ("inst_dep_m", ">u4"),
        ("SP80", ">u1"),
        ("beam", ">u1"),
        ("SP81", ">u1"),
        ("volt_scale", ">u1"),
        ("SP146", ">u1"),
        ("f", ">u4"),
        ("SP83", ">u1"),
        ("unknown_83", ">u1"),
        ("SP84", ">u1"),
        ("unknown_84", ">u1"),
        ("SP149", ">u1"),
        ("unknown_149", ">u4"),
        ("SP86", ">u1"),
        ("e_err_m", ">u1"),
        ("SP87", ">u1"),
        ("n_err_m", ">u1"),
        ("SP160", ">u1"),
        ("ping_cnt", ">u4"),
        ("head_end", ">u1"),
    ]
)

HEADER_STRUCTURE_152 = np.dtype(
    [
        ("head_start", ">u4"),
        ("SP128", ">u1"),
        ("record_num", ">u4"),
        ("SP129", ">u1"),
        ("time_s", ">u4"),
        ("SP130", ">u1"),
        ("utm_e", ">i4"),
        ("SP131", ">u1"),
        ("utm_n", ">i4"),
        ("SP132", ">u1"),
        ("gps1", ">u2"),
        ("instr_heading", ">u2"),
        ("SP133", ">u1"),
        ("gps2", ">u2"),
        ("speed_ms", ">u2"),
        ("SP134", ">u1"),
        ("unknown_134", ">u4"),
        ("SP135", ">u1"),
        ("inst_dep_m", ">u4"),
        ("SP136", ">u1"),
        ("unknown_136", ">u4"),
        ("SP137", ">u1"),
        ("unknown_137", ">u4"),
        ("SP138", ">u1"),
        ("unknown_138", ">u4"),
        ("SP139", ">u1"),
        ("unknown_139", ">u4"),
        ("SP140", ">u1"),
        ("unknown_140", ">u4"),
        ("SP141", ">u1"),
        ("unknown_141", ">u4"),
        ("SP142", ">u1"),
        ("unknown_142", ">u4"),
        ("SP143", ">u1"),
        ("unknown_143", ">u4"),
        ("SP80", ">u1"),
        ("beam", ">u1"),
        ("SP81", ">u1"),
        ("volt_scale", ">u1"),
        ("SP146", ">u1"),
        ("f", ">u4"),
        ("SP83", ">u1"),
        ("unknown_83", ">u1"),
        ("SP84", ">u1"),
        ("unknown_84", ">u1"),
        ("SP149", ">u1"),
        ("unknown_149", ">u4"),
        ("SP86", ">u1"),
        ("e_err_m", ">u1"),
        ("SP87", ">u1"),
        ("n_err_m", ">u1"),
        ("SP152", ">u1"),
        ("unknown_152", ">u4"),
        ("SP153", ">u1"),
        ("unknown_153", ">u4"),
        ("SP154", ">u1"),
        ("unknown_154", ">u4"),
        ("SP155", ">u1"),
        ("unknown_155", ">u4"),
        ("SP156", ">u1"),
        ("unknown_156", ">u4"),
        ("SP157", ">u1"),
        ("unknown_157", ">u4"),
        ("SP158", ">u1"),
        ("unknown_158", ">u4"),
        ("SP159", ">u1"),
        ("unknown_159", ">u4"),
        ("SP160", ">u1"),
        ("ping_cnt", ">u4"),
        ("head_end", ">u1"),
    ]
)

SONAR_HEADER_REGISTRY: dict[int, np.dtypes.VoidDType] = {
    67: HEADER_STRUCTURE_67,
    72: HEADER_STRUCTURE_72,
    152: HEADER_STRUCTURE_152,
}


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
class Header(Protocol): ...


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


class SonarData:
    def __init__(self): ...


@dataclass
class Data:
    metadata: Metadata | None = None
    beam_data: list[pd.DataFrame] | None = None


def read_data(file: Path, metadata_to_csv: bool) -> Data:
    metadata = read_metadata(file)
    beam_data = read_sonar_data(metadata, metadata_to_csv)
    data = Data(metadata=metadata, beam_data=beam_data)
    return data


def read_sonar_data(metadata: Metadata, metadata_to_csv: bool = False) -> np.ndarray:
    beams = get_beams(metadata.file_path)

    for beam in beams:
        df = read_all_pings(beam.file, metadata)
        if metadata_to_csv:
            beam.output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(beam.output_file, index=False)
        beam.metadata = df

    return beams


def read_all_pings(
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
            header_data, new_pointer = read_single_ping(
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
    df = _doUnitConversion(
        df, metadata.pixel_size, metadata.temperature, metadata.unix_time
    )

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

    # Save to csv - TODO: Refactor this outside this function.
    # header_data_all.to_csv(out_file, index=False)

    return df


def read_single_ping(
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


def get_head_structure(header_length: int) -> np.dtypes.VoidDType:
    """Returns an instance of the appropriate header dataclass."""
    header_structure = SONAR_HEADER_REGISTRY.get(header_length)
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


def _doUnitConversion(
    df: pd.DataFrame, pixel_size: float, temperature: float, unix_time: int
) -> pd.DataFrame:
    """ """

    # Calculate range
    df["max_range"] = df["ping_cnt"] * pixel_size

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

    # Reproject latitude/longitude to UTM zone
    e, n = convert_latlon_to_localutm(
        latitudes=lat,
        longitudes=lon,
        utm_zone=infer_utm_zone(lon.mean()),
    )
    df["e"] = e
    df["n"] = n

    # Instrument heading, speed, and depth need to be divided by 10 to be
    # in appropriate units.
    df["instr_heading"] = df["instr_heading"] / 10
    df["speed_ms"] = df["speed_ms"] / 10
    df["inst_dep_m"] = df["inst_dep_m"] / 10

    # Get units into appropriate format
    df["f"] = df["f"] / 1000  # Hertz to Kilohertz
    df["time_s"] = df["time_s"] / 1000  # milliseconds to seconds
    df["tempC"] = temperature * 10
    # # Can we figure out a way to base transducer length on where we think the recording came from?
    # df['t'] = 0.108
    # Use recording unix time to calculate each sonar records unix time
    try:
        # starttime = float(df['unix_time'])
        starttime = float(unix_time)
        df["caltime"] = starttime + df["time_s"]

    except:
        df["caltime"] = 0

    # Update caltime to timestamp
    sonTime = []
    sonDate = []
    needToFilt = False
    for t in df["caltime"].to_numpy():
        try:
            t = datetime.datetime.fromtimestamp(t)
            sonDate.append(datetime.datetime.date(t))
            sonTime.append(datetime.datetime.time(t))
        except:
            sonDate.append(-1)
            sonTime.append(-1)
            needToFilt = True

    df = df.drop("caltime", axis=1)
    df["date"] = sonDate
    df["time"] = sonTime

    if needToFilt:
        df = df[df["date"] != -1]
        df = df[df["time"] != -1]

        df = df.dropna()

    df = df[df["e"] != np.inf]
    df = df[df["record_num"] >= 0]

    lastIdx = df["index"].iloc[-1]
    df = df[df["index"] <= lastIdx]

    # Calculate along-track distance from 'time's and 'speed_ms'. Approximate distance estimate
    df = _calcTrkDistTS(df)

    # Add transect number (for aoi processing)
    df["transect"] = 0

    return df


def _calcTrkDistTS(df: pd.DataFrame) -> pd.DataFrame:
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


def beam_files(path: Path):
    stem = path.stem
    return (path.parent / stem).rglob("*.SON")


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


def infer_utm_zone(longitude: float) -> int:
    """
    Infers the UTM zone from a given longitude.

    Parameters:
    - longitude (float): Longitude in decimal degrees.

    Returns:
    - int: UTM zone number.
    """
    return int((longitude + 180) / 6) + 1


def convert_latlon_to_localutm(
    latitudes: Sequence, longitudes: Sequence, utm_zone: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts latitude and longitude to local UTM coordinates.

    Parameters:
    - latitudes (array-like): Latitude values in decimal degrees.
    - longitudes (array-like): Longitude values in decimal degrees.
    - utm_zone (int): UTM zone number.

    Returns:
    - (numpy.ndarray, numpy.ndarray): (easting, northing) in meters.
    """
    # Define the UTM projection for the specified zone
    proj = pyproj.Proj(proj="utm", zone=utm_zone, ellps="WGS84", preserve_units=True)

    # Perform the transformation
    easting, northing = proj(longitudes, latitudes)

    return np.array(easting), np.array(northing)
