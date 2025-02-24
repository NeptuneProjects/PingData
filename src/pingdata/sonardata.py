# -*- coding: utf-8 -*-

from dataclasses import dataclass
import datetime
from enum import StrEnum
import os
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from pingdata.metadata import read_metadata
from pingdata.fread import fread_data


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
class Header67:
    head_start: int
    SP128: int
    record_num: int
    SP129: int
    time_s: int
    SP130: int
    utm_e: int
    SP131: int
    utm_n: int
    SP132: int
    gps1: int
    instr_heading: int
    SP133: int
    gps2: int
    speed_ms: int
    SP135: int
    inst_dep_m: int
    SP80: int
    beam: int
    SP81: int
    volt_scale: int
    SP146: int
    f: int
    SP83: int
    unknown_83: int
    SP84: int
    unknown_84: int
    SP149: int
    unknown_149: int
    SP86: int
    e_err_m: int
    SP87: int
    n_err_m: int
    SP160: int
    ping_cnt: int
    head_end: int


@dataclass
class Header72:
    head_start: int
    SP128: int
    record_num: int
    SP129: int
    time_s: int
    SP130: int
    utm_e: int
    SP131: int
    utm_n: int
    SP132: int
    gps1: int
    instr_heading: int
    SP133: int
    gps2: int
    speed_ms: int
    SP134: int
    unknown_134: int
    SP135: int
    inst_dep_m: int
    SP80: int
    beam: int
    SP81: int
    volt_scale: int
    SP146: int
    f: int
    SP83: int
    unknown_83: int
    SP84: int
    unknown_84: int
    SP149: int
    unknown_149: int
    SP86: int
    e_err_m: int
    SP87: int
    n_err_m: int
    SP160: int
    ping_cnt: int
    head_end: int


@dataclass
class Header152:
    head_start: int
    SP128: int
    record_num: int
    SP129: int
    time_s: int
    SP130: int
    utm_e: int
    SP131: int
    utm_n: int
    SP132: int
    gps1: int
    instr_heading: int
    SP133: int
    gps2: int
    speed_ms: int
    SP134: int
    unknown_134: int
    SP135: int
    inst_dep_m: int
    SP136: int
    unknown_136: int
    SP137: int
    unknown_137: int
    SP138: int
    unknown_138: int
    SP139: int
    unknown_139: int
    SP140: int
    unknown_140: int
    SP141: int
    unknown_141: int
    SP142: int
    unknown_142: int
    SP143: int
    unknown_143: int
    SP80: int
    beam: int
    SP81: int
    volt_scale: int
    SP146: int
    f: int
    SP83: int
    unknown_83: int
    SP84: int
    unknown_84: int
    SP149: int
    unknown_149: int
    SP86: int
    e_err_m: int
    SP87: int
    n_err_m: int
    SP152: int
    unknown_152: int
    SP153: int
    unknown_153: int
    SP154: int
    unknown_154: int
    SP155: int
    unknown_155: int
    SP156: int
    unknown_156: int
    SP157: int
    unknown_157: int
    SP158: int
    unknown_158: int
    SP159: int
    unknown_159: int
    SP160: int
    ping_cnt: int
    head_end: int


HEADER_REGISTRY: dict[int, Header] = {
    67: Header67,
    72: Header72,
    152: Header152,
}


def get_head_struct(header_length: int, **kwargs) -> Header:
    """Returns an instance of the appropriate header dataclass."""
    header_cls = HEADER_REGISTRY.get(header_length)
    if not header_cls:
        raise ValueError(f"Unsupported header size: {header_length}")
    return header_cls(**kwargs)


def _getBeamName(self, beam: str) -> str:
    return BeamType.from_beam(beam).value


def ping_header_length(file: Path, head_end_val: int = 33) -> int:
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

    with file.open("rb") as f:
        i = 0  # Counter to track sonar header length
        foundEnd = False  # Flag to track if end of sonar header found
        while foundEnd is False and i < 200:
            lastPos = (
                f.tell()
            )  # Get current position in file (byte offset from beginning)
            byte = fread_data(f, 1, "B")  # Decode byte value

            # Check if we found the end of the ping.
            ## A value of 33 **may** indicate end of ping.
            if byte[0] == head_end_val and lastPos > 3:
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
            i += 1

    # i reaches 200, then we have exceeded known Humminbird header length.
    ## Set i to 0, then the next sonFile will be checked.
    if i == 200:
        i = 0

    # headBytes = i # Store data in class attribute for later use
    return i


def read_sonardata(file: Path):
    metadata = read_metadata(file)

    return metadata


def _parsePingHeader(in_file: str, out_file: str):
    """ """

    # Get file length
    file_len = os.path.getsize(in_file)

    # Initialize counter
    i = 0
    chunk_i = 0
    chunk = 0

    header_dat_all = []

    frame_offset = []

    chunk_id = []

    file = open(in_file, "rb")

    # Decode ping header
    while i < file_len:

        # Get header data at offset i
        header_dat, cpos = _getPingHeader(file, i)

        # Add frame offset
        frame_offset.append(i)

        header_dat_all.append(header_dat)

        chunk_id.append(chunk)

        # update counter with current position
        i = cpos

        if chunk_i == nchunk:
            chunk_i = 0
            chunk += 1
        else:
            chunk_i += 1

    header_dat_all = pd.DataFrame.from_dict(header_dat_all)

    # Add in the frame offset
    header_dat_all["index"] = frame_offset

    # Add in the son_offset (headBytes for Humminbird)
    header_dat_all["son_offset"] = headBytes

    # Add chunk id
    header_dat_all["chunk_id"] = chunk_id

    # Do unit conversions
    header_dat_all = _doUnitConversion(header_dat_all)

    # Drop spacer and unknown columns
    for col in header_dat_all.columns:
        if "SP" in col:
            header_dat_all.drop(col, axis=1, inplace=True)

        if not exportUnknown and "unknown" in col:
            header_dat_all.drop(col, axis=1, inplace=True)

    # Drop head_start
    header_dat_all.drop("head_start", axis=1, inplace=True)
    header_dat_all.drop("head_end", axis=1, inplace=True)

    # Update last chunk if too small (for rectification)
    lastChunk = header_dat_all[header_dat_all["chunk_id"] == chunk]
    if len(lastChunk) <= nchunk / 2:
        header_dat_all.loc[header_dat_all["chunk_id"] == chunk, "chunk_id"] = chunk - 1

    # Save to csv
    header_dat_all.to_csv(out_file, index=False)

    return


def _getPingHeader(file, i: int):
    # Get necessary attributes
    head_struct = son_struct
    length = frame_header_size  # Account for start and end header

    # Move to offset
    file.seek(i)

    # Get the data
    buffer = file.read(length)

    # Read the data
    header = np.frombuffer(buffer, dtype=head_struct)

    out_dict = {}
    for name, typ in header.dtype.fields.items():
        out_dict[name] = header[name][0].item()

    # Next ping header is from current position + ping_cnt
    next_ping = file.tell() + header[0][-2]

    return out_dict, next_ping


def _doUnitConversion(self, df: pd.DataFrame):
    """ """

    # Calculate range
    df["max_range"] = df["ping_cnt"] * pixM

    # Easting and northing variances appear to be stored in the file
    ## They are reported in cm's so need to convert
    df["e_err_m"] = np.abs(df["e_err_m"]) / 100
    df["n_err_m"] = np.abs(df["n_err_m"]) / 100

    # Now calculate hdop from n/e variances
    df["hdop"] = np.round(np.sqrt(df["e_err_m"] + df["n_err_m"]), 2)

    # Convert eastings/northings to latitude/longitude (from Py3Hum - convert using International 1924 spheroid)
    lat = (
        np.arctan(
            np.tan(np.arctan(np.exp(df["utm_n"] / 6378388.0)) * 2.0 - 1.570796326794897)
            * 1.0067642927
        )
        * 57.295779513082302
    )
    lon = (df["utm_e"] * 57.295779513082302) / 6378388.0

    df["lon"] = lon
    df["lat"] = lat

    # Reproject latitude/longitude to UTM zone
    e, n = trans(lon, lat)
    df["e"] = e
    df["n"] = n

    # Instrument heading, speed, and depth need to be divided by 10 to be
    ## in appropriate units.
    df["instr_heading"] = df["instr_heading"] / 10
    df["speed_ms"] = df["speed_ms"] / 10
    df["inst_dep_m"] = df["inst_dep_m"] / 10

    # Get units into appropriate format
    df["f"] = df["f"] / 1000  # Hertz to Kilohertz
    df["time_s"] = df["time_s"] / 1000  # milliseconds to seconds
    df["tempC"] = tempC * 10
    # # Can we figure out a way to base transducer length on where we think the recording came from?
    # df['t'] = 0.108
    # Use recording unix time to calculate each sonar records unix time
    try:
        # starttime = float(df['unix_time'])
        starttime = float(humDat["unix_time"])
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

    # Other corrections Dan did, not implemented yet...
    # if sonHead['beam']==3 or sonHead['beam']==2:
    #     dist = ((np.tan(25*0.0174532925))*sonHead['inst_dep_m']) +(tvg)
    #     bearing = 0.0174532925*sonHead['instr_heading'] - (pi/2)
    #     bearing = bearing % 360
    #     sonHead['heading'] = bearing
    # print("\n\n", sonHead, "\n\n")

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
