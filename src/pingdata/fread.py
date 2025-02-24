# -*- coding: utf-8 -*-

from array import array as arr
from typing import BinaryIO


def fread_data(binary_data: BinaryIO, num_bytes: int, byte_type: str):
    data = arr(byte_type)
    data.fromfile(binary_data, num_bytes)
    return list(data)
