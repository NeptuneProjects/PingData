# -*- coding: utf-8 -*-

from dataclasses import dataclass, field

import numpy as np


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

PING_HEADER_STRUCTURE_67 = np.dtype(
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
        ("f_kHz", ">u4"),
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

PING_HEADER_STRUCTURE_72 = np.dtype(
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
        ("f_kHz", ">u4"),
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

PING_HEADER_STRUCTURE_152 = np.dtype(
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
        ("f_kHz", ">u4"),
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

PING_HEADER_REGISTRY: dict[int, np.dtypes.VoidDType] = {
    67: PING_HEADER_STRUCTURE_67,
    72: PING_HEADER_STRUCTURE_72,
    152: PING_HEADER_STRUCTURE_152,
}


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
